#hepps_dataset.py

import os
from collections import defaultdict

import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from scipy.signal import find_peaks, butter, sosfiltfilt, resample
from enum import Enum
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
        self.labels = []
        self.original_lengths = []
        self.pwv_by_label = defaultdict(list)

        # Process all files upon initialization.
        self._process_all_files()

    def _load_all_data(self):
        self.data_list = []
        try:
            file_pattern = os.path.join(self.root, 'finger_vs_wrist*.txt')
            txt_files = glob.glob(file_pattern)

            for file_path in txt_files:
                try:
                    data = np.loadtxt(file_path, skiprows=6)
                    filename = os.path.basename(file_path)
                    label = filename.rsplit('_', 1)[-1].replace('.txt', '')

                    print(f"Loaded {file_path} with shape {data.shape}")

                    self.data_list.append({
                        'label': label,
                        'timestamps': data[:, 0],
                        'wrist': data[:, 2],
                        'finger': data[:, 1]
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        except Exception as e:
            print(f"Failed to scan directory {self.root}: {e}")

        # Create label-to-integer mapping using Enum
        unique_labels = sorted(set(entry['label'] for entry in self.data_list))
        self.LabelEnum = Enum('LabelEnum', {label: idx for idx, label in enumerate(unique_labels)})

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
    
    def _ensure_alternating(self, peaks, troughs):
        """Return a list of trough indices that alternate with peaks."""
        combined = sorted([(p, 'peak') for p in peaks] +
                        [(t, 'trough') for t in troughs],
                        key=lambda x: x[0])

        alternating = []
        for idx, typ in combined:
            if not alternating or alternating[-1][1] != typ:
                alternating.append((idx, typ))

        while alternating and alternating[0][1] != 'trough':
            alternating.pop(0)

        return [idx for idx, typ in alternating if typ == 'trough']
    
    def _extract_segments(
            self,
            label: str,
            wrist_sig: np.ndarray,
            finger_sig: np.ndarray,
            troughs: list[int],
            do_resample: bool = True      
    ):
        """
        Cut wrist/finger signals into segments bracketed by consecutive troughs.

        Parameters
        ----------
        label : str
            Condition label for every segment in this file.
        wrist_sig, finger_sig : np.ndarray
            Filtered 1‑D signals for wrist and finger.
        troughs : list[int]
            Indices of trough positions (already alternating / cleaned).
        do_resample : bool, default True
            If True, resample every segment to `self.target_length`.
            If False, keep the raw segment length.
        """
        for i in range(len(troughs) - 1):
            interval = troughs[i + 1] - troughs[i]
            dyn_off  = int(self.segment_offset * interval)
            s = max(troughs[i]   - dyn_off, 0)
            e = min(troughs[i+1] - dyn_off, len(wrist_sig))

            w_seg = wrist_sig[s:e]
            f_seg = finger_sig[s:e]
            resample_factor = len(w_seg) / self.target_length

            if do_resample:
                w_seg = self._resample_signal(w_seg)
                f_seg = self._resample_signal(f_seg)

            w_seg = self._normalize_signal(w_seg, self.normalize_wrist_factor)
            f_seg = self._normalize_signal(f_seg, self.normalize_finger_factor)

            self.segments.append(np.stack((w_seg, f_seg), axis=-1))
            self.labels.append(label)
            self.original_lengths.append(resample_factor)

    def _remove_outliers(self):
        """3‑sigma filter per label; updates self.segments / self.labels."""
        grouped = defaultdict(list)
        for seg, lab in zip(self.segments, self.labels):
            grouped[lab].append(seg)

        keep_seg, keep_lab = [], []
        for lab, segs in grouped.items():
            arr = np.array(segs)                 # (N,L,2)
            mean = arr.mean(axis=0)              # (L,2)
            std  = arr.std(axis=0)
            low, high = mean - 3*std, mean + 3*std

            for seg in segs:
                if np.all((seg >= low) & (seg <= high)):
                    keep_seg.append(seg)
                    keep_lab.append(lab)

        self.segments, self.labels = keep_seg, keep_lab
        print(f"Outlier removal: retained {len(keep_seg)} segments.")

    def _process_all_files(self):
        self._load_all_data()                    
        for file in self.data_list:         
            label   = file['label']
            wrist   = file['wrist']            
            finger  = file['finger']
            timestamps = file['timestamps']
            wrist = self._apply_butterworth(wrist,  self.filter_cutoff_wrist)
            finger = self._apply_butterworth(finger, self.filter_cutoff_finger)

            detect = finger if self.sensor_type == 'finger' else wrist
            peaks, _   = self._find_peaks( detect,          self.distance_sec)
            troughs,_  = self._find_peaks(-detect,          self.distance_sec)

            clean_troughs = self._ensure_alternating(peaks, troughs)
            self._extract_segments(label, wrist, finger, clean_troughs)
        self._remove_outliers()

    # ------------------------------------------------------------------
    #  PUBLIC: compute raw PWV (no filtering, no resampling)
    # ------------------------------------------------------------------
    def compute_pwv(self, max_delay_samples: int = 10_000) -> dict:
        """
        Return {label: [δ₁, δ₂, …]} where each δ is the wrist→finger delay
        **in seconds**.  A pair is kept if 0 < delay < max_delay_samples.
        """
        pwv_by_label = defaultdict(list)

        for entry in self.data_list:
            label  = entry["label"]
            wrist  = entry["wrist"]
            finger = entry["finger"]

            f_peaks, _ = self._find_peaks(finger, self.distance_sec)
            w_peaks, _ = self._find_peaks(wrist,  self.distance_sec)

            combined = sorted([(i, "finger") for i in f_peaks] +
                              [(i, "wrist")  for i in w_peaks],
                              key=lambda x: x[0])

            alternating = []
            for idx, typ in combined:
                if not alternating or alternating[-1][1] != typ:
                    alternating.append((idx, typ))

            i = 0
            while i < len(alternating) - 1:
                idx1, typ1 = alternating[i]
                idx2, typ2 = alternating[i + 1]

                if typ1 == "wrist" and typ2 == "finger":
                    delta_samples = idx2 - idx1
                    if 0 < delta_samples < max_delay_samples:
                        # -------- convert to seconds -------------
                        delta_sec = delta_samples / self.sampling_rate
                        pwv_by_label[label].append(delta_sec)
                    i += 2
                else:
                    i += 1

        return pwv_by_label


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
    # Instantiate the dataset using your full segment extraction mode
    dataset = HePPSDataset(
        segment_offset=0,
        target_length=1024,
        filter_cutoff_wrist=[0.5, 20], 
        filter_cutoff_finger=[0.2, 30],
        remove_after_resample=10
    )

    idx = 0
    print(dataset.segments[idx].shape)
    print(dataset.labels[idx])

    # plt.plot(dataset.segments[idx])
    # plt.show()

    pwv_dict = dataset.compute_pwv()
    for lab, deltas in pwv_dict.items():
        print(f"{lab}:  n={len(deltas)}  mean delay={np.mean(deltas):.5f} seconds")