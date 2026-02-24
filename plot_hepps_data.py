#!/usr/bin/env python3
"""
Plot HEPPS data from CSV file with timestamp, ain0, ain1 columns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

FILE_PATH = r'data\HEPPSBP\Changxin\20260218_125104.csv'

def filter_data_percentage(data, threshold=0.005):
    """
    Filter data to replace points that are more than threshold smaller than previous point.
    Uses the filtered (replaced) value for the next comparison to handle consecutive outliers.
    
    Args:
        data (pd.Series): Input data series
        threshold (float): Threshold as decimal (0.005 = 0.5%, 0.01 = 1%, etc.)
    
    Returns:
        pd.Series: Filtered data
    """
    filtered_data = data.copy()
    
    for i in range(1, len(data)):
        current_val = filtered_data.iloc[i]  # Use current value from filtered data
        prev_val = filtered_data.iloc[i-1]   # Use previous value from filtered data
        
        # Check if current value is more than threshold smaller than previous
        if prev_val > 0:  # Avoid division by zero
            relative_decrease = (prev_val - current_val) / prev_val
            if relative_decrease > threshold:
                filtered_data.iloc[i] = prev_val
                
    return filtered_data

# Backward compatibility functions
def filter_data_ain0(data, threshold=0.005):
    """Backward compatibility wrapper for AIN0 filtering"""
    return filter_data_percentage(data, threshold)

def filter_data_ain1(data, threshold=0.01):
    """Backward compatibility wrapper for AIN1 filtering - now uses percentage threshold"""
    return filter_data_percentage(data, threshold)

def apply_butterworth_filter(data, cutoff_freq=20, sampling_rate=180, order=2):
    """
    Apply a mild Butterworth low-pass filter to the data
    
    Args:
        data (pd.Series): Input data series
        cutoff_freq (float): Cutoff frequency in Hz (default: 10 Hz)
        sampling_rate (float): Sampling rate in Hz (default: 180 Hz)
        order (int): Filter order (default: 2 for mild filtering)
    
    Returns:
        pd.Series: Filtered data
    """
    # Normalize the cutoff frequency
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    return pd.Series(filtered_data, index=data.index)

def plot_hepps_data(csv_path):
    """
    Plot HEPPS data with ain0 and ain1 in separate subplots
    
    Args:
        csv_path (str): Path to the CSV file containing timestamp, ain0, ain1 columns
    """
    # Read the CSV data
    try:
        data = pd.read_csv(csv_path)
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data range: {data['timestamp'].min():.3f}s to {data['timestamp'].max():.3f}s")
        print(f"AIN0 range: {data['ain0'].min()} to {data['ain0'].max()}")
        print(f"AIN1 range: {data['ain1'].min()} to {data['ain1'].max()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create figure with 2 subplots (vertical layout)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('HEPPS Data - AIN0 and AIN1 Channels', fontsize=16, fontweight='bold')
    
    # Plot AIN0 channel
    ax1.plot(data['timestamp'], data['ain0'], color='tab:red', linewidth=1, alpha=0.8)
    ax1.set_title('AIN0 Channel (Wrist Sensor)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ADC Value')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(data['timestamp'].min(), data['timestamp'].max())
    
    # Add statistics text for AIN0
    ain0_stats = f"Mean: {data['ain0'].mean():.1f}, Std: {data['ain0'].std():.1f}, Range: [{data['ain0'].min()}, {data['ain0'].max()}]"
    ax1.text(0.02, 0.98, ain0_stats, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    # Plot AIN1 channel
    ax2.plot(data['timestamp'], data['ain1'], color='tab:blue', linewidth=1, alpha=0.8)
    ax2.set_title('AIN1 Channel (Finger Sensor)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('ADC Value')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(data['timestamp'].min(), data['timestamp'].max())
    
    # Add statistics text for AIN1
    ain1_stats = f"Mean: {data['ain1'].mean():.1f}, Std: {data['ain1'].std():.1f}, Range: [{data['ain1'].min()}, {data['ain1'].max()}]"
    ax2.text(0.02, 0.98, ain1_stats, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print additional information
    duration = data['timestamp'].max() - data['timestamp'].min()
    sample_rate = len(data) / duration
    print(f"\nRecording duration: {duration:.1f} seconds")
    print(f"Number of samples: {len(data)}")
    print(f"Approximate sample rate: {sample_rate:.1f} Hz")

def plot_hepps_overlay(csv_path):
    """
    Plot HEPPS data with ain0 and ain1 overlaid on the same plot
    
    Args:
        csv_path (str): Path to the CSV file containing timestamp, ain0, ain1 columns
    """
    # Read the CSV data
    data = pd.read_csv(csv_path)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    fig.suptitle('HEPPS Data - AIN0 and AIN1 Channels Overlaid', fontsize=16, fontweight='bold')
    
    # Plot both channels
    ax.plot(data['timestamp'], data['ain0'], color='tab:red', linewidth=1, alpha=0.7, label='AIN0 (Wrist)')
    ax.plot(data['timestamp'], data['ain1'], color='tab:blue', linewidth=1, alpha=0.7, label='AIN1 (Finger)')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('ADC Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(data['timestamp'].min(), data['timestamp'].max())
    
    plt.tight_layout()
    plt.show()

def plot_hepps_segment(csv_path, start_time=0, duration=10):
    """
    Plot a specific time segment of HEPPS data with filtering
    
    Args:
        csv_path (str): Path to the CSV file
        start_time (float): Start time in seconds
        duration (float): Duration to plot in seconds
    """
    data = pd.read_csv(csv_path)    # Apply filtering to both channels with percentage thresholds
    ain0_threshold = 0.018  # 0.5% for AIN0 (wrist)
    ain1_threshold = 0.01   # 1.0% for AIN1 (finger)
    
    print(f"Applying percentage-based filtering - AIN0: {ain0_threshold*100:.1f}%, AIN1: {ain1_threshold*100:.1f}%...")
    data['ain0_filtered'] = filter_data_percentage(data['ain0'], threshold=ain0_threshold)
    data['ain1_filtered'] = filter_data_percentage(data['ain1'], threshold=ain1_threshold)
    
    # Apply Butterworth filter on top of the outlier filtering
    print("Applying Butterworth low-pass filter (10 Hz cutoff)...")
    data['ain0_butterworth'] = apply_butterworth_filter(data['ain0_filtered'], cutoff_freq=10, sampling_rate=180)
    data['ain1_butterworth'] = apply_butterworth_filter(data['ain1_filtered'], cutoff_freq=10, sampling_rate=180)
    
    # Count how many points were filtered
    ain0_filtered_count = (data['ain0'] != data['ain0_filtered']).sum()
    ain1_filtered_count = (data['ain1'] != data['ain1_filtered']).sum()
    
    print(f"AIN0: {ain0_filtered_count} points filtered out of {len(data)} ({ain0_filtered_count/len(data)*100:.2f}%)")
    print(f"AIN1: {ain1_filtered_count} points filtered out of {len(data)} ({ain1_filtered_count/len(data)*100:.2f}%)")
    
    # Filter data for the specified time range
    end_time = start_time + duration
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    segment_data = data[mask]
    
    if len(segment_data) == 0:
        print(f"No data found in time range {start_time}s to {end_time}s")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'HEPPS Data Segment - {start_time}s to {end_time}s (Filtered)', fontsize=16, fontweight='bold')
      # Plot AIN0 channel
    ax1.plot(segment_data['timestamp'], segment_data['ain0'], color='tab:red', linewidth=1, alpha=0.3, label='Original')
    ax1.plot(segment_data['timestamp'], segment_data['ain0_filtered'], color='tab:red', linewidth=1, alpha=0.6, label='Outlier Filtered')
    ax1.plot(segment_data['timestamp'], segment_data['ain0_butterworth'], color='darkred', linewidth=2, label='Butterworth Filtered')
    ax1.set_title('AIN0 Channel (Wrist Sensor)', fontsize=14)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ADC Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot AIN1 channel
    ax2.plot(segment_data['timestamp'], segment_data['ain1'], color='tab:blue', linewidth=1, alpha=0.3, label='Original')
    ax2.plot(segment_data['timestamp'], segment_data['ain1_filtered'], color='tab:blue', linewidth=1, alpha=0.6, label='Outlier Filtered')
    ax2.plot(segment_data['timestamp'], segment_data['ain1_butterworth'], color='darkblue', linewidth=2, label='Butterworth Filtered')
    ax2.set_title('AIN1 Channel (Finger Sensor)', fontsize=14)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('ADC Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define the path to the HEPPS data file
    csv_file = FILE_PATH
    
    print("Plotting HEPPS data segment (25-28 seconds)...")
    print("=" * 50)

    # Plot a 3-second segment from 25-28 seconds
    plot_hepps_segment(csv_file, start_time=23, duration=5)
