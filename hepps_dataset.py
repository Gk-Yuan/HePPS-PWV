#hepps_dataset.py

import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks, butter, sosfiltfilt, resample
import matplotlib.pyplot as plt

class HePPSDataset(Dataset):
    """
    A PyTorch Dataset for HePPS data.
    This class loads txt files (skipping the first 6 rows) that contain three columns:
       - timestamps
       - wrist sensor data
       - fingertip sensor data
       
    For each file (specified by a file identifier such as "rest"), it:
      1. Constructs the full filename using the root directory (default: './data') and a fixed filename pattern.
      2. Loads the file and extracts the timestamps, wrist, and finger data.
      3. Detects peaks (using the specified sensor for detection) via an internal _find_peaks method.
      4. Extracts full segments from both sensors using a configurable offset from the detected peaks.
      
    Each segment is a 2D array of shape (L, 2), where L (which may vary) is the number of samples
    between the defined segment boundaries. The condition label is derived from the file identifier.
    """

    def __init__(self, 
                 file_identifiers, 
                 root='./data', 
                 sampling_rate=100000, 
                 target_length=80000,
                 remove_after_resample=500,
                 sensor_type='finger',
                 distance_sec=0.4, 
                 prominence=0.005, 
                 segment_offset=0, 
                 filter_order=3,
                 filter_cutoff_finger=[0.2, 30],
                 filter_cutoff_wrist=[0.5, 20],
                 normalize_finger_factor=0.02,
                 normalize_wrist_factor=0.002,
                 transform=None):
        """
        Parameters:
            file_identifiers (list of str): List of file identifiers 
                                            (e.g., ["rest", "exercise", "deep-breath", ...]).
            root (str): Directory where all data files are stored.
            sampling_rate (int): Sampling rate in Hz (default is 100000).
            sensor_type (str): Which sensor signal to use for peak detection ("finger" or "wrist").
            distance_sec (float): Minimum time (in seconds) between consecutive peaks.
            prominence (float): Minimum required prominence for peaks.
            segment_offset (float): Time offset (in seconds) to subtract from each peak index when defining segments.
            transform (callable, optional): A transformation to apply on each sample.
        """
        self.file_identifiers = file_identifiers
        self.root = root
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.remove_after_resample = remove_after_resample
        self.sensor_type = sensor_type.lower()  # used for peak detection
        self.distance_sec = distance_sec
        self.segment_offset = segment_offset  # in seconds
        self.prominence = prominence
        self.filter_order = filter_order
        self.filter_cutoff_finger = filter_cutoff_finger
        self.filter_cutoff_wrist = filter_cutoff_wrist
        self.transform = transform
        self.normalize_finger_factor = normalize_finger_factor
        self.normalize_wrist_factor = normalize_wrist_factor

        self.segments = []  # List to store full segments (each a 2D array with shape [L, 2])
        self.labels = []    # List to store corresponding condition labels
        self.original_lengths = []

        # Process all files upon initialization.
        self._process_all_files()

    def _load_file(self, file_identifier):
        """
        Loads a text file using np.loadtxt, skipping the first 6 rows.
        Constructs the full file path by joining self.root with the filename pattern:
            'finger_vs_wrist0403_' + file_identifier + '.txt'
        Extracts and returns a dictionary with:
            - 'timestamps': first column,
            - 'wrist': second column,
            - 'finger': third column.
        """
        try:
            file_path = os.path.join(self.root, 'finger_vs_wrist0403_' + file_identifier + '.txt')
            data = np.loadtxt(file_path, skiprows=6)
            print(f"Loaded {file_path} with shape {data.shape}")
            return {
                'timestamps': data[:, 0],
                'wrist': data[:, 2],
                'finger': data[:, 1]
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _find_peaks(self, signal, distance_sec):
        """
        Internal method to detect peaks in a 1D signal using SciPy's find_peaks.
        
        Parameters:
            signal (np.array): 1D array representing the sensor signal.
            distance_sec (float): Minimum time (in seconds) between consecutive peaks.
            prominence (float): Minimum required prominence for a peak.
        
        Returns:
            peaks (np.array): Indices of the detected peaks.
            properties (dict): Additional properties from find_peaks.
        """
        distance_samples = int(distance_sec * self.sampling_rate)
        peaks, properties = find_peaks(signal, distance=distance_samples, prominence=self.prominence)
        return peaks, properties
    
    def _apply_butterworth(self, sensor_data, filter_cutoff):
        """
        Apply a butterworth filter to the sensor data.
        """
        sos = butter(self.filter_order, filter_cutoff,
                     btype='bandpass', fs=self.sampling_rate, output='sos', analog=False)
        filtered_data = sosfiltfilt(sos, sensor_data, axis=0)
        return filtered_data
    
    def _resample_signal(self, segment):
        """
        Resample the signal to target_length and scale the trigger offset accordingly.
        Returns the resampled signal and the scaled trigger offset.
        """
        resampled_signal = resample(segment, self.target_length + self.remove_after_resample, axis=0)
        resampled_signal_remove = resampled_signal[self.remove_after_resample//2: self.remove_after_resample//2 + self.target_length]
        return resampled_signal_remove
    
    def _normalize_signal(self, segment, factor):
        normalized_signal = segment / factor
        return normalized_signal

    def _process_all_files(self):
        """
        Processes each file from self.file_identifiers:
        1. Loads the file using _load_file, which returns a dictionary containing timestamps,
            wrist, and finger data.
        2. Applies the Butterworth filter to both wrist and finger signals.
        3. Selects one sensor (specified by self.sensor_type) for peak detection.
        4. Detects peaks and troughs (using the prominence parameter) and enforces an alternating pattern.
        5. Converts the segment offset from seconds to samples.
        6. Extracts full segments from both sensor channels between (trough - offset) and (next trough - offset)
            and resamples them.
        7. After all segments (for a given condition) are collected, groups segments by condition and, for each group,
            computes the per-timepoint mean and std (with the assumption that every segment has been resampled to a fixed length).
            Any segment with any value outside [mean - 3·std, mean + 3·std] (computed per time index and per channel) is discarded.
        """
        # For each file (identified by its file_identifier, e.g., "rest", "exercise", etc.)
        for file_identifier in self.file_identifiers:
            data_dict = self._load_file(file_identifier)
            if data_dict is None:
                continue

            # Load raw data.
            timestamps = data_dict['timestamps']
            wrist_signal = data_dict['wrist']
            finger_signal = data_dict['finger']

            # Apply Butterworth filtering to both signals.
            filtered_wrist_signal = self._apply_butterworth(wrist_signal, self.filter_cutoff_wrist)
            filtered_finger_signal = self._apply_butterworth(finger_signal, self.filter_cutoff_finger)

            # Use the chosen sensor for detection.
            detection_signal = filtered_finger_signal if self.sensor_type == 'finger' else filtered_wrist_signal
            negative_detection_signal = -detection_signal 

            # Detect peaks and troughs using the internal _find_peaks method with the prominence.
            peaks, _ = self._find_peaks(detection_signal, self.distance_sec)
            troughs, _ = self._find_peaks(negative_detection_signal, self.distance_sec)
            print(f"File {file_identifier}: Detected {len(peaks)} peaks and {len(troughs)} troughs.")

            # Combine the detections into one sorted list with their type labels.
            combined = sorted([(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs], key=lambda x: x[0])
            alternating = []
            for idx, typ in combined:
                if not alternating or alternating[-1][1] != typ:
                    alternating.append((idx, typ))
                else:
                    continue
            while alternating and alternating[0][1] != 'trough':
                alternating.pop(0)
            filtered_troughs = [pos for pos, typ in alternating if typ == 'trough']

            # Optionally store intermediate results for further inspection.
            self.wrist = filtered_wrist_signal
            self.finger = filtered_finger_signal
            self.troughs = filtered_troughs
            self.peaks = peaks

            # Convert segment offset from seconds to samples.
            offset_samples = int(self.segment_offset * self.sampling_rate)

            parts = file_identifier.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                label = parts[0]
            else:
                label = file_identifier

            # Extract segments from one trough to the next.
            for i in range(len(filtered_troughs) - 1):
                # Compute the dynamic offset based on the interval between consecutive troughs.
                interval_samples = filtered_troughs[i+1] - filtered_troughs[i]
                dynamic_offset = int(self.segment_offset * interval_samples)
                start_idx = max(filtered_troughs[i] - dynamic_offset, 0)
                end_idx = min(filtered_troughs[i+1] - dynamic_offset, len(detection_signal))
                wrist_segment = filtered_wrist_signal[start_idx:end_idx]
                finger_segment = filtered_finger_signal[start_idx:end_idx]
                original_len = len(wrist_segment)
                resampled_factor = original_len / self.target_length
                resampled_wrist = self._resample_signal(wrist_segment)
                resampled_finger = self._resample_signal(finger_segment)
                normalize_wrist = self._normalize_signal(resampled_wrist, self.normalize_wrist_factor)
                normalize_finger = self._normalize_signal(resampled_finger, self.normalize_finger_factor)
                # Combine both channels into a 2D segment with shape (L, 2)
                segment = np.stack((normalize_wrist, normalize_finger), axis=-1)
                self.segments.append(segment)
                self.labels.append(label)
                self.original_lengths.append(resampled_factor)

        # --- Outlier removal by condition ---
        # Group segments by label.
        grouped_segments = defaultdict(list)
        grouped_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            grouped_segments[label].append(self.segments[idx])

        filtered_segments = []
        filtered_labels = []
        for label, seg_list in grouped_segments.items():
            seg_array = np.array(seg_list)  # shape: (N_segments, L, 2)
            # Compute per-timepoint mean and std for each channel, for the given condition.
            mean_wrist = np.mean(seg_array[:, :, 0], axis=0)  # shape: (L,)
            std_wrist = np.std(seg_array[:, :, 0], axis=0)
            mean_finger = np.mean(seg_array[:, :, 1], axis=0)
            std_finger = np.std(seg_array[:, :, 1], axis=0)
            # For each segment, check whether every data point is within [mean - 3*std, mean + 3*std]
            for seg in seg_list:
                wrist_ok = np.all((seg[:, 0] >= mean_wrist - 3*std_wrist) & (seg[:, 0] <= mean_wrist + 3*std_wrist))
                finger_ok = np.all((seg[:, 1] >= mean_finger - 3*std_finger) & (seg[:, 1] <= mean_finger + 3*std_finger))
                if wrist_ok and finger_ok:
                    filtered_segments.append(seg)
                    filtered_labels.append(label)
        print(f"Outlier removal: retained {len(filtered_segments)} out of {len(self.segments)} segments.")
        self.segments = filtered_segments
        self.labels = filtered_labels

    def __len__(self):
        """Return the number of extracted segments."""
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Retrieve the segment and its corresponding label as a PyTorch tensor.
        Each segment is a 2D array with shape (L, 2), where L may vary.
        """
        sample = self.segments[idx]           # [L,2] resampled waveform
        length = self.original_lengths[idx]   # int
        label  = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        # normalize length by the target_length if you like:
        length = torch.tensor(length / self.target_length, 
                            dtype=torch.float32).unsqueeze(0)
        return sample, length, label

if __name__ == '__main__':
    # Define file paths and parameters (make sure these paths are correct)
    file_identifiers = [
        "rest",
        "caffeine",
        "deep_breath",
        "hold_breath_1",
        "hold_breath_2",
        "exercise"
    ]

    # Instantiate the dataset using your full segment extraction mode
    dataset = HePPSDataset(
        file_identifiers=file_identifiers, 
        segment_offset=0,
        target_length=1024,
        filter_cutoff_wrist=[0.5, 20], 
        filter_cutoff_finger=[0.2, 30],
        remove_after_resample=10
    )

    idx = 0
    print(dataset.segments[idx].shape)
    print(dataset.labels[idx])

    plt.plot(dataset.segments[idx])
    plt.show()