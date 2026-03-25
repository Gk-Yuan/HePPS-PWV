# hepps_bp_dataset.py
import os, glob
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks, butter, sosfiltfilt, resample, filtfilt, savgol_filter
import matplotlib.pyplot as plt

class HEPPSBPDataset(Dataset):
    """
    Dataset for the HePPS repository with new structure:

    └── ./data/HEPPSBP/
        ├── ParticipantName1/
        │   ├── 20251105_130421.csv
        │   └── 20251106_142315.csv
        ├── ParticipantName2/
        │   └── 20251105_135542.csv
        └── metadata_super.csv   (columns: Name, time, SBP1, DBP1, ...)

    * Each .csv has three columns: timestamp, ain0, ain1
    * The folder name is the participant name/label
    * SBP/DBP values are looked up from metadata_super.csv via (Name, time)
      where time matches each CSV filename stem (YYYYMMDD_HHMMSS)
    * Signals are **interpolated** onto a uniform grid (`target_fs` Hz)
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(self,
                 root='./data/HEPPSBP/',
                 target_fs=350,              # Hz after interpolation
                 distance_sec=0.4,
                 prominence=0.005,
                 filter_order=3,             # Restored original bandpass filter order
                 filter_cutoff_wrist=[0.5, 10],    # Bandpass filter cutoffs for wrist [low, high]
                 filter_cutoff_finger=[0.2, 10],   # Bandpass filter cutoffs for finger [low, high]
                 sensor='finger',            # Which sensor to use for peak/trough detection ('finger' or 'wrist')
                 segment_offset=0.0,
                 target_length=256,
                 remove_after_resample=15,                 
                 normalize_finger_factor=1e5,
                 normalize_wrist_factor=1e5,
                 outlier_threshold_wrist=0.018,
                 outlier_threshold_finger=0.01,
                 savgol_window_length=11,
                 savgol_polyorder=3,
                 peak_index_min=30,
                 peak_index_max=100,
                 require_both_peak_range=False,
                 bp_sample_ratio=0.05,       # sample BP within value +/- (value * ratio)
                 peak_distance_max=50,       # max allowed |wrist_peak - finger_peak| (samples)
                 interpolation_length=None,     # Length for interpolation after resampling (None = no interpolation)
                 transform=None):
        self.root  = root
        self.fs    = target_fs            # uniform sampling rate (Hz)
        self.distance_sec = distance_sec  # for peak detection
        self.prominence   = prominence
        self.filter_order = filter_order
        self.filter_cutoff_wrist = filter_cutoff_wrist   # Bandpass filter cutoffs for wrist [low, high]
        self.filter_cutoff_finger = filter_cutoff_finger # Bandpass filter cutoffs for finger [low, high]
        self.sensor = sensor.lower()        # Which sensor to use for peak/trough detection
        self.segment_offset = segment_offset        
        self.target_len     = target_length
        self.rm_after       = remove_after_resample
        self.norm_finger_fac= normalize_finger_factor
        self.norm_wrist_fac = normalize_wrist_factor
        self.outlier_threshold_wrist = outlier_threshold_wrist     # Outlier filtering threshold for wrist
        self.outlier_threshold_finger = outlier_threshold_finger   # Outlier filtering threshold for finger
        self.savgol_window_length = int(savgol_window_length)
        self.savgol_polyorder = int(savgol_polyorder)
        self.peak_index_min = int(peak_index_min)
        self.peak_index_max = int(peak_index_max)
        self.require_both_peak_range = self._to_bool(require_both_peak_range)
        self.bp_sample_ratio = bp_sample_ratio
        self.peak_distance_max = peak_distance_max
        self.interpolation_length = interpolation_length           # Length for interpolation after resampling
        self.transform = transform

        # containers ----------------------------------------------------
        self.raw_list   = []   # raw signals after interpolation
        self.segments   = []   # (L,2) wrist-finger
        self.labels     = []   # string label
        self.vpwv       = []   # ground-truth vPWV from CSV
        self.sbp_list   = []
        self.dbp_list   = []

        # --------------------------------------------------------------

        self._load_all_files()
        self._extract_all_segments()


    # ------------------------------------------------------------------
    # file scan  +  interpolation
    # ------------------------------------------------------------------
    def _load_all_files(self):
        # --- load metadata csv with SBP/DBP columns --------------------
        csv_path = os.path.join(self.root, 'metadata_super.csv')
        meta = pd.read_csv(
            csv_path,
            usecols=['Name', 'time', 'SBP1', 'DBP1']
        )
        # Build lookup dict: (Name, time) -> (SBP1, DBP1)
        self.bp_map = {}
        for _, row in meta.iterrows():
            name = str(row['Name']).strip().lower()
            session_time = str(row['time']).strip()
            sbp = pd.to_numeric(row['SBP1'], errors='coerce')
            dbp = pd.to_numeric(row['DBP1'], errors='coerce')
            self.bp_map[(name, session_time)] = (sbp, dbp)

        # --- scan participant folders for *.csv files ------------------
        # New structure: data/HEPPSBP/ParticipantName/YYYYMMDD_HHMMSS.csv
        participant_folders = [d for d in os.listdir(self.root) 
                              if os.path.isdir(os.path.join(self.root, d)) and d != '__pycache__']
        
        csv_paths = []
        for participant in participant_folders:
            participant_path = os.path.join(self.root, participant)
            # Find all CSV files in this participant's folder
            participant_csvs = glob.glob(os.path.join(participant_path, '*.csv'))
            csv_paths.extend(participant_csvs)
        
        if not csv_paths:
            raise RuntimeError(
                f'No csv files found in participant sub-folders of {self.root}'
            )

        for path in csv_paths:
            try:
                raw = pd.read_csv(path)
                # Extract participant name from folder name and time from file name
                participant_name = os.path.basename(os.path.dirname(path))
                session_time = os.path.splitext(os.path.basename(path))[0]
                sbp_vals, dbp_vals = self.bp_map.get((participant_name.lower(), session_time), (np.nan, np.nan))
                if not np.isfinite(float(sbp_vals)) or not np.isfinite(float(dbp_vals)):
                    print(
                        f"Warning: No metadata match for Name='{participant_name}', "
                        f"time='{session_time}' in metadata_super.csv"
                    )
                
                t      = raw['timestamp'].values    # seconds
                wrist  = raw['ain1'].values         # ain1 is wrist
                finger = raw['ain0'].values         # ain0 is finger

                # --- interpolate onto uniform grid --------------------
                new_t = np.arange(t[0], t[-1], 1/self.fs)
                wrist_i  = np.interp(new_t, t, wrist)
                finger_i = np.interp(new_t, t, finger)
                self.raw_list.append({
                    'label' : participant_name,
                    't'     : new_t,
                    'wrist' : wrist_i,
                    'finger': finger_i,
                    'sbp_vals': sbp_vals,
                    'dbp_vals': dbp_vals
                })
            except Exception as e:
                print(f'skip {path}: {e}')

        # label enumeration (optional)
        # uniq = sorted({d['label'] for d in self.raw_list})
        # self.LabelEnum = Enum('LabelEnum', {lab: i for i, lab in enumerate(uniq)})


    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_bp(self, value):
        """
        Sample a random BP from value Â± (value * self.bp_sample_ratio).
        """
        bp = float(value)
        if not np.isfinite(bp):
            return bp

        ratio = abs(float(self.bp_sample_ratio))
        lower = bp * (1.0 - ratio)
        upper = bp * (1.0 + ratio)
        if lower > upper:
            lower, upper = upper, lower

        return float(np.random.uniform(lower, upper))
    
    def _filter_outliers(self, data, threshold=0.005):
        """
        Filter data to replace points that are more than threshold smaller than previous point.
        Uses the filtered (replaced) value for the next comparison to handle consecutive outliers.
        
        Args:
            data (np.ndarray): Input data array
            threshold (float): Threshold as decimal (0.005 = 0.5%, 0.01 = 1.0%)
        
        Returns:
            np.ndarray: Filtered data
        """
        filtered_data = data.copy()
        
        for i in range(1, len(data)):
            current_val = filtered_data[i]  # Use current value from filtered data
            prev_val = filtered_data[i-1]   # Use previous value from filtered data
            
            # Check if current value is more than threshold smaller than previous
            if prev_val > 0:  # Avoid division by zero
                relative_decrease = (prev_val - current_val) / prev_val
                if relative_decrease > threshold:
                    filtered_data[i] = prev_val
                    
        return filtered_data

    @staticmethod
    def _to_bool(value):
        """
        Parse common bool representations robustly (e.g., "true"/"false", 1/0).
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {'1', 'true', 't', 'yes', 'y', 'on'}:
                return True
            if v in {'0', 'false', 'f', 'no', 'n', 'off', ''}:
                return False
        return bool(value)
    
    def _savgol(self, x):
        """
        Apply Savitzky-Golay smoothing with safe adjustment for short signals.
        """
        n = len(x)
        if n < 3:
            return x

        win = min(self.savgol_window_length, n)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return x

        poly = min(self.savgol_polyorder, win - 1)
        if poly < 1:
            return x

        return savgol_filter(x, window_length=win, polyorder=poly)

    def _butter(self, x, filter_cutoff):
        """
        Apply Butterworth bandpass filter with specified cutoff frequencies
        
        Args:
            x: Signal to filter
            filter_cutoff: [low_freq, high_freq] cutoff frequencies
        """
        nyquist_freq = self.fs / 2
        low_norm = filter_cutoff[0] / nyquist_freq
        high_norm = filter_cutoff[1] / nyquist_freq
        b, a = butter(self.filter_order, [low_norm, high_norm], btype='bandpass', analog=False)
        return filtfilt(b, a, x)

    def _find(self, x):               # wrapper: return peak indices
        dist = int(self.distance_sec * self.fs)        
        idx, _ = find_peaks(x, distance=dist, prominence=self.prominence)
        return idx

    def _resample(self, x):
        y = resample(x, self.target_len + self.rm_after)
        return y[self.rm_after//2 : self.rm_after//2 + self.target_len]
    
    def _interpolate(self, x):
        """
        Interpolate signal to specified length using linear interpolation.
        """
        if self.interpolation_length is None:
            return x
        
        original_indices = np.linspace(0, len(x) - 1, len(x))
        target_indices = np.linspace(0, len(x) - 1, self.interpolation_length)
        return np.interp(target_indices, original_indices, x)
    
    def _ensure_alternating(self, peaks_a, peaks_b, tag_a='w', tag_b='f'):
        """
        Merge two peak lists and keep only strictly alternating entries.
        Returns a list of (idx, tag) tuples.
        """
        merged = sorted([(i, tag_a) for i in peaks_a] +
                        [(i, tag_b) for i in peaks_b], key=lambda x: x[0])

        altern = []
        for idx, typ in merged:
            if not altern or altern[-1][1] != typ:
                altern.append((idx, typ))
        return altern

    def _is_valid_peak_distance(self, seg_w, seg_f):
        """
        Keep segment only when wrist and finger peak locations are close enough.
        """
        wrist_peaks = self._find(seg_w)
        finger_peaks = self._find(seg_f)
        if len(wrist_peaks) == 0 or len(finger_peaks) == 0:
            return False

        # Minimum distance between any wrist peak and any finger peak
        min_dist = np.min(np.abs(wrist_peaks[:, None] - finger_peaks[None, :]))
        return bool(min_dist <= self.peak_distance_max)

    # def _normalize_by_label(self):
    #     """
    #     Normalize segments per label and per channel: for each label, scale all its segments
    #     so that the maximum absolute value for each channel equals 1.
    #     """
    #     # compute max abs for each label and each channel across its segments
    #     max_per_label = {}
    #     for lab in set(self.labels):
    #         # indices for this label
    #         idxs = [i for i, l in enumerate(self.labels) if l == lab]
    #         # max abs per channel among those segments
    #         max_w = max(np.max(np.abs(self.segments[i][:, 0])) for i in idxs)
    #         max_f = max(np.max(np.abs(self.segments[i][:, 1])) for i in idxs)
    #         # guard against zero or non-finite
    #         if not np.isfinite(max_w) or max_w == 0:
    #             max_w = 1.0
    #         if not np.isfinite(max_f) or max_f == 0:
    #             max_f = 1.0
    #         max_per_label[lab] = (max_w, max_f)

    #     # apply per-label, per-channel scaling
    #     scaled = []
    #     for seg, lab in zip(self.segments, self.labels):
    #         max_w, max_f = max_per_label[lab]
    #         seg_scaled = seg.copy()
    #         seg_scaled[:, 0] = seg_scaled[:, 0] / max_w
    #         seg_scaled[:, 1] = seg_scaled[:, 1] / max_f
    #         scaled.append(seg_scaled)
    #     self.segments = scaled

    def _normalize_and_flip_segments(self):
        """
        For each segment and each channel (wrist, finger):
        1) min-max normalize to [-1, 1]
        2) flip sign so -1 -> 1 and 1 -> -1
        """
        def norm_flip_channel(x):
            arr = np.asarray(x, dtype=np.float64)
            if arr.size == 0:
                return arr
            mask = np.isfinite(arr)
            if not np.any(mask):
                return arr
            vmin = np.min(arr[mask])
            vmax = np.max(arr[mask])
            out = arr.copy()
            if vmax > vmin:
                out[mask] = 2.0 * (arr[mask] - vmin) / (vmax - vmin) - 1.0
            else:
                out[mask] = 0.0
            out[mask] = -out[mask]
            return out

        normalized = []
        for seg in self.segments:
            seg_arr = np.asarray(seg, dtype=np.float64).copy()
            seg_arr[:, 0] = norm_flip_channel(seg_arr[:, 0])  # wrist
            seg_arr[:, 1] = norm_flip_channel(seg_arr[:, 1])  # finger
            normalized.append(seg_arr)
        self.segments = normalized

    def _keep_segments_with_peak_in_range(self):
        """
        Keep only segments whose main peak index is within [peak_index_min, peak_index_max].
        If require_both_peak_range=True, both wrist and finger must pass.
        Otherwise, only the channel selected by `self.sensor` is checked.
        """
        if not self.segments:
            return

        channel_idx = 1 if self.sensor == 'finger' else 0
        lo = self.peak_index_min
        hi = self.peak_index_max
        if lo > hi:
            lo, hi = hi, lo

        kept_idx = []
        for i, seg in enumerate(self.segments):
            seg_arr = np.asarray(seg)
            if self.require_both_peak_range:
                wrist_sig = seg_arr[:, 0]
                finger_sig = seg_arr[:, 1]
                wrist_peaks = self._find(wrist_sig)
                finger_peaks = self._find(finger_sig)
                if len(wrist_peaks) == 0 or len(finger_peaks) == 0:
                    continue

                wrist_main_peak = int(wrist_peaks[np.argmax(wrist_sig[wrist_peaks])])
                finger_main_peak = int(finger_peaks[np.argmax(finger_sig[finger_peaks])])
                if (lo <= wrist_main_peak <= hi) and (lo <= finger_main_peak <= hi):
                    kept_idx.append(i)
            else:
                sig = seg_arr[:, channel_idx]
                peaks = self._find(sig)
                if len(peaks) == 0:
                    continue

                main_peak = int(peaks[np.argmax(sig[peaks])])
                if lo <= main_peak <= hi:
                    kept_idx.append(i)

        start_n = len(self.segments)
        self.segments = [self.segments[i] for i in kept_idx]
        self.labels = [self.labels[i] for i in kept_idx]
        self.sbp_list = [self.sbp_list[i] for i in kept_idx]
        self.dbp_list = [self.dbp_list[i] for i in kept_idx]
        print(f"Peak-index filter [{lo}, {hi}]: kept {len(self.segments)}/{start_n} segments.")
        # ------------------------------------------------------------------
    # segment extraction with alternating enforcement + normalisation
    # ------------------------------------------------------------------      
    def _extract_all_segments(self):
        for rec in self.raw_list:
            # Step 1: Apply outlier filtering with configurable thresholds
            wrist_filtered = self._filter_outliers(rec['wrist'], threshold=self.outlier_threshold_wrist)
            finger_filtered = self._filter_outliers(rec['finger'], threshold=self.outlier_threshold_finger)
            
            # Step 2: Apply Savitzky-Golay smoothing before Butterworth filter
            wrist_smoothed = self._savgol(wrist_filtered)
            finger_smoothed = self._savgol(finger_filtered)
            
            # Step 3: Apply Butterworth filter
            w_filt = self._butter(wrist_smoothed, self.filter_cutoff_wrist)
            f_filt = self._butter(finger_smoothed, self.filter_cutoff_finger)
            
            # Step 4: Resample the filtered signals
            wrist_resampled = resample(w_filt, int(len(w_filt) * 0.8))  # Example resampling
            finger_resampled = resample(f_filt, int(len(f_filt) * 0.8))
            
            # Step 5: Interpolate if interpolation_length is specified
            wrist_interp = self._interpolate(wrist_resampled)
            finger_interp = self._interpolate(finger_resampled)
            
            # Use processed signals for further processing
            w_processed = wrist_interp
            f_processed = finger_interp# Step 4: Use the specified sensor for peak/trough detection
            if self.sensor == 'finger':
                detect_signal = f_processed
            elif self.sensor == 'wrist':
                detect_signal = w_processed
            else:
                raise ValueError(f"Invalid sensor '{self.sensor}'. Must be 'finger' or 'wrist'.")
            
            tr   = self._find(-detect_signal)    # troughs
            pk   = self._find(detect_signal)     # peaks
            alt  = self._ensure_alternating(pk, tr, 'p', 't')
            peaks = [idx for idx, typ in alt if typ == 'p']
            if len(peaks) < 2:
                continue

            for a, b in zip(peaks[:-1], peaks[1:]):
                seg_w = w_processed[a:b]
                seg_f = f_processed[a:b]
                if len(seg_w) < 32:
                    continue
                seg_w = self._resample(seg_w)
                seg_f = self._resample(seg_f)
                if not self._is_valid_peak_distance(seg_w, seg_f):
                    continue

                segment = np.stack((seg_w, seg_f), -1)

                # Keep original metadata targets (SBP1/DBP1) without jitter.
                sbp = float(rec['sbp_vals'])
                dbp = float(rec['dbp_vals'])
                self.segments.append(segment)
                self.labels.append(rec['label'])
                self.sbp_list.append(sbp) 
                self.dbp_list.append(dbp)

        # Store original segments before outlier removal for visualization
        self.original_segments = self.segments.copy()
        self.original_labels = self.labels.copy()
        print(f"Total segments before outlier removal: {len(self.original_segments)}")
        
        self._remove_outliers()           # <<< 3-Ïƒ filtering at the end
        self._normalize_and_flip_segments()
        self._keep_segments_with_peak_in_range()
        # self._normalize_by_label()

    # ------------------------------------------------------------------
    # 3-sigma outlier removal per label
    # ------------------------------------------------------------------
    def _remove_outliers(self):
        """
        Remove segments that are >3Ïƒ from the mean, computed separately for each label.
        """
        # group indices (not raw segments) by label
        grouped_idx = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            grouped_idx[lab].append(idx)

        num_seg = len(self.segments)

        kept_idx = []
        # for each label, compute Î¼ and Ïƒ over its own segments
        for lab, idxs in grouped_idx.items():
            arr = np.stack([self.segments[i] for i in idxs], axis=0)  # shape (N, L, 2)
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
            lo, hi = mu - 3*sd, mu + 3*sd

            # only keep indices whose segment lies within [lo, hi]
            for i in idxs:
                seg = self.segments[i]
                if np.all((seg >= lo) & (seg <= hi)):
                    kept_idx.append(i)

        # rebuild all parallel lists based on kept indices
        self.segments = [self.segments[i] for i in kept_idx]
        self.labels   = [self.labels[i]   for i in kept_idx]
        self.sbp_list = [self.sbp_list[i] for i in kept_idx]
        self.dbp_list = [self.dbp_list[i] for i in kept_idx]

        print(f"Outlier removal: started with {num_seg} segments, "
            f"kept {len(self.segments)} segments.")
        
    # ------------------------------------------------------------------
    # PWV (raw, unfiltered)  â€” same logic as earlier but time-based
    # ------------------------------------------------------------------
    def compute_pwv(self, max_delay=0.10):
        out = defaultdict(list)
        for rec in self.raw_list:
            label = rec['label']
            ts = rec['t']
            f_pk = self._find(rec['finger'])
            w_pk = self._find(rec['wrist'])
            comb = sorted([(i,'f') for i in f_pk] + [(i,'w') for i in w_pk])
            alt  = []
            for idx,t in comb:
                if not alt or alt[-1][1]!=t: alt.append((idx,t))

            i = 0
            while i < len(alt)-1:
                i1,t1 = alt[i]
                i2,t2 = alt[i+1]
                if t1=='w' and t2=='f':
                    dt = (ts[i2]-ts[i1])/self.fs
                    if 0<dt<max_delay:
                        out[rec['label']].append(dt)
                    i += 2
                else:
                    i += 1
            print(label, len(w_pk), len(f_pk), len(out[label]))

        return out

    # ------------------------------------------------------------------
    # torch-dataset boilerplate
    # ------------------------------------------------------------------
    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        x   = torch.tensor(self.segments[idx], dtype=torch.float32)
        # y   = self.LabelEnum[self.labels[idx]].value
        sbp = torch.tensor(self.sbp_list[idx], dtype=torch.float32)
        dbp = torch.tensor(self.dbp_list[idx], dtype=torch.float32) 
        if self.transform:
            x = self.transform(x)
        return x, sbp, dbp
    
if __name__ == '__main__':
    ds = HEPPSBPDataset()
    print(len(ds), "segments")
    
    # Continue with regular analysis
    idx = 0
    print(f"\nFirst segment shape: {ds.segments[idx].shape}")
    print(f"First segment label: {ds.labels[idx]}")
    x, sbp, dbp = ds[idx]
    print("sbp:", sbp, "dbp: ", dbp)

    # plt.plot(ds.segments[idx])
    # plt.show()
    # pwv = ds.compute_pwv()
    # for lab, d in pwv.items():
    #     print(lab, "mean dt =", np.mean(d), "s")

    label_to_plot = 'gongkai'
    
    # Collect all segments for that label
    segments = [seg for seg, lab in zip(ds.segments, ds.labels) if lab == label_to_plot]
    
    if segments:  # Only plot if we have segments for this label
        # Plot wrist channel for all segments
        plt.figure()
        for seg in segments:
            plt.plot(seg[:, 0], alpha=0.3, c='lightgray')
        avg_segment = np.mean(np.stack(segments, axis=0), axis=0)
        plt.plot(avg_segment[:, 0], lw=2, c='tab:red')
        plt.title(f"All wrist segments for {label_to_plot}")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized amplitude")
        plt.show()
        
        # Plot finger channel for all segments
        plt.figure()
        for seg in segments:
            plt.plot(seg[:, 1], alpha=0.3, c='lightgray')
        avg_segment = np.mean(np.stack(segments, axis=0), axis=0)
        plt.plot(avg_segment[:, 1], lw=2, c='tab:red')    
        plt.title(f"All finger segments for {label_to_plot}")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized amplitude")
        plt.show()

        # Overlay wrist and finger (original + average) on the same plot
        plt.figure()
        for seg in segments:
            plt.plot(seg[:, 0], alpha=0.25, c='lightcoral')  # wrist originals
            plt.plot(seg[:, 1], alpha=0.25, c='lightblue')   # finger originals
        avg_segment = np.mean(np.stack(segments, axis=0), axis=0)
        plt.plot(avg_segment[:, 0], lw=2.5, c='darkred', label='Wrist avg')
        plt.plot(avg_segment[:, 1], lw=2.5, c='darkblue', label='Finger avg')
        plt.title(f"Wrist + Finger overlay for {label_to_plot}")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized amplitude")
        plt.legend()
        plt.show()

        # 3) Overlay SBP & DBP values
        sbp_vals = [sbp for sbp, lab in zip(ds.sbp_list, ds.labels) if lab == label_to_plot]
        dbp_vals = [dbp for dbp, lab in zip(ds.dbp_list, ds.labels) if lab == label_to_plot]
        
        # plt.figure(figsize=(6,4))
        # plt.plot(sbp_vals, marker='o', linestyle='-', label='SBP', color='tab:red')
        # plt.plot(dbp_vals, marker='x', linestyle='--', label='DBP', color='tab:blue')
        # plt.title(f"SBP vs DBP for {label_to_plot}")
        # plt.xlabel("Segment index")
        # plt.ylabel("Pressure (mmHg)")
        # plt.legend()
        # plt.show()
    else:
        print(f"No segments found for participant: {label_to_plot}")
