# cml_dataset.py
import os, glob
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks, butter, sosfiltfilt, resample
import matplotlib.pyplot as plt

class CMLDataset(Dataset):
    """
    Dataset for the CML repository

    └── ./data/CML/
        ├── 2024-02-09/
        │   ├── Andy.txt
        │   └── Bella.txt
        ├── 2024-03-12/
        │   └── Andy.txt
        └── metadata_super.csv   (columns: Name , vPWV , …)

    * Each .txt has three columns: time-stamp  wrist  finger
    * The file name (before .txt) is the label, e.g. 'Andy'
    * vPWV is looked up from metadata_super.csv via the Name column
    * Signals are **interpolated** onto a uniform grid (`target_fs` Hz)
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(self,
                 root='./data/CML',
                 target_fs=1_000,              # Hz after interpolation
                 distance_sec=0.4,
                 prominence=0.005,
                 filter_order=3,
                 filter_cutoff_finger=(0.2, 30),
                 filter_cutoff_wrist =(0.5, 20),
                 segment_offset=0.0,
                 target_length=8192,
                 remove_after_resample=128,
                 normalize_finger_factor=1e5,
                 normalize_wrist_factor =1e5,
                 transform=None):
        self.root  = root
        self.fs    = target_fs            # uniform sampling rate (Hz)
        self.distance_sec = distance_sec  # for peak detection
        self.prominence   = prominence
        self.filter_order = filter_order
        self.cut_finger   = filter_cutoff_finger
        self.cut_wrist    = filter_cutoff_wrist
        self.segment_offset = segment_offset
        self.target_len     = target_length
        self.rm_after       = remove_after_resample
        self.norm_finger_fac= normalize_finger_factor
        self.norm_wrist_fac = normalize_wrist_factor
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
            usecols=['Name', 'SBP1', 'SBP2', 'SBP3', 'DBP1', 'DBP2', 'DBP3']
        )
        # build lookup dicts: Name -> [SBP1, SBP2, SBP3]
        self.sbp_map = {
            row['Name']: [row['SBP1'], row['SBP2'], row['SBP3']]
            for _, row in meta.iterrows()
        }
        # build lookup dicts: Name -> [DBP1, DBP2, DBP3]
        self.dbp_map = {
            row['Name']: [row['DBP1'], row['DBP2'], row['DBP3']]
            for _, row in meta.iterrows()
        }

        # --- recursive glob for *.txt ----------------------------------
        txt_paths = glob.glob(os.path.join(self.root, '*', '*.txt'))
        if not txt_paths:
            raise RuntimeError(
                f'No txt files found in sub-folders of {self.root}'
            )

        for path in txt_paths:
            try:
                raw = np.loadtxt(path, delimiter=',')
                label = os.path.splitext(os.path.basename(path))[0]
                sbp_vals = self.sbp_map.get(label, [np.nan]*3)
                dbp_vals = self.dbp_map.get(label, [np.nan]*3)
                t    = raw[:, 0]            # seconds
                wrist= raw[:, 2]
                finger=raw[:, 1]

                # --- interpolate onto uniform grid --------------------
                new_t = np.arange(t[0], t[-1], 1/self.fs)
                wrist_i  = np.interp(new_t, t, wrist)
                finger_i = np.interp(new_t, t, finger)
                self.raw_list.append({
                    'label' : label,
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

    def _sample_bp(self, values):
        """
        Sample a random BP between (min(values) - sd) and (max(values) + sd),
        with guards against non-finite or overflow bounds.
        """
        arr = np.array(values, dtype=np.float64)
        low, high = np.nanmin(arr), np.nanmax(arr)      
        sd = np.nanstd(arr)                             
        lower, upper = low - sd, high + sd         

        # ensure bounds are finite
        if not np.isfinite(lower):
            lower = low
        if not np.isfinite(upper):
            upper = high

        # if range is invalid, fall back to mean
        if lower >= upper:
            return float(arr.mean())

        try:
            return float(np.random.uniform(lower, upper))
        except OverflowError:
            return float(arr.mean())
    
    def _butter(self, x, band):
        sos = butter(self.filter_order, band, 'band', fs=self.fs, output='sos')
        return sosfiltfilt(sos, x, axis=0)

    def _find(self, x):               # wrapper: return peak indices
        dist = int(self.distance_sec * self.fs)
        idx, _ = find_peaks(x, distance=dist, prominence=self.prominence)
        return idx

    def _resample(self, x):
        y = resample(x, self.target_len + self.rm_after)
        return y[self.rm_after//2 : self.rm_after//2 + self.target_len]
    
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
    
    def _normalize_by_label(self):
        """
        Normalize segments per label: for each label, scale all its segments
        so that the maximum absolute value across all segments equals 1.
        """
        # CHANGED: compute max abs for each label across its segments
        max_per_label = {}
        for lab in set(self.labels):
            # indices for this label
            idxs = [i for i, l in enumerate(self.labels) if l == lab]
            # global max abs value among those segments
            max_val = max(np.max(np.abs(self.segments[i])) for i in idxs)
            # guard against zero or non-finite
            if not np.isfinite(max_val) or max_val == 0:
                max_val = 1.0
            max_per_label[lab] = max_val

        # CHANGED: apply per-label scaling
        self.segments = [
            seg / max_per_label[lab]
            for seg, lab in zip(self.segments, self.labels)
        ]

    # ------------------------------------------------------------------
    # segment extraction with alternating enforcement + normalisation
    # ------------------------------------------------------------------
    def _extract_all_segments(self):
        for rec in self.raw_list:
            w_filt = self._butter(rec['wrist'],  self.cut_wrist)
            f_filt = self._butter(rec['finger'], self.cut_finger)

            # enforce finger-trough alternation against finger peaks
            tr   = self._find(-f_filt)          # troughs
            pk   = self._find(f_filt)           # peaks
            alt  = self._ensure_alternating(pk, tr, 'p', 't')
            troughs = [idx for idx, typ in alt if typ == 't']
            if len(troughs) < 2:
                continue

            for a, b in zip(troughs[:-1], troughs[1:]):
                seg_w = w_filt[a:b]
                seg_f = f_filt[a:b]
                if len(seg_w) < 32:
                    continue

                seg_w = self._resample(seg_w)
                seg_f = self._resample(seg_f)
                
                segment = np.stack((seg_w, seg_f), -1)

                sbp = self._sample_bp(rec['sbp_vals']) 
                dbp = self._sample_bp(rec['dbp_vals'])
                self.segments.append(segment)
                self.labels.append(rec['label'])
                self.sbp_list.append(sbp) 
                self.dbp_list.append(dbp)

        self._remove_outliers()           # <<< 3-σ filtering at the end
        self._normalize_by_label()

    # ------------------------------------------------------------------
    # 3-sigma outlier removal per label
    # ------------------------------------------------------------------
    def _remove_outliers(self):
        """
        Remove segments that are >3σ from the mean, computed separately for each label.
        """
        # group indices (not raw segments) by label
        grouped_idx = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            grouped_idx[lab].append(idx)

        num_seg = len(self.segments)

        kept_idx = []
        # for each label, compute μ and σ over its own segments
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
    # PWV (raw, unfiltered)  — same logic as earlier but time-based
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
    ds = CMLDataset(target_fs=1000, target_length=4096)
    print(len(ds), "segments")
    idx = 0
    print(ds.segments[idx].shape)
    print(ds.labels[idx])
    x, label, sbp, dbp = ds[idx]
    print("sbp:", sbp, "dbp: ", dbp)

    # plt.plot(ds.segments[idx])
    # plt.show()
    # pwv = ds.compute_pwv()
    # for lab, d in pwv.items():
    #     print(lab, "mean dt =", np.mean(d), "s")


    label_to_plot = 'Erik'
    
    # Collect all segments for that label
    segments = [seg for seg, lab in zip(ds.segments, ds.labels) if lab == label_to_plot]
    
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

    # 3) Overlay SBP & DBP values
    sbp_vals = [sbp for sbp, lab in zip(ds.sbp_list, ds.labels) if lab == label_to_plot]
    dbp_vals = [dbp for dbp, lab in zip(ds.dbp_list, ds.labels) if lab == label_to_plot]
    
    plt.figure(figsize=(6,4))
    plt.plot(sbp_vals, marker='o', linestyle='-', label='SBP', color='tab:red')
    plt.plot(dbp_vals, marker='x', linestyle='--', label='DBP', color='tab:blue')
    plt.title(f"SBP vs DBP for {label_to_plot}")
    plt.xlabel("Segment index")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.show()