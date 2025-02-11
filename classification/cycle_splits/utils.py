import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
# Function to compute keypoints relative to the hip position
def compute_relative_keypoints(keypoints, ref_index):
    """Compute keypoints relative to the hip keypoint."""
    hip_x, hip_y, hip_v = keypoints[ref_index * 3 : ref_index * 3 + 3]
    
    if hip_v == 0:
        return [(None, None)] * (len(keypoints) // 3)

    relative_keypoints = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]
        if v > 0:
            rel_x = x - hip_x
            rel_y = y - hip_y
        else:
            rel_x = rel_y = None
        relative_keypoints.append((rel_x, rel_y))
    
    return relative_keypoints

# Function to normalize signals using Z-score normalization
def normalize_signal(signal):
    """Normalize the signal using Z-score normalization."""
    signal = np.array(signal, dtype=np.float32)
    valid_values = signal[~np.isnan(signal)]

    if len(valid_values) > 1:
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        signal = (signal - mean) / std if std > 0 else signal - mean
    return signal

# Function to apply Gaussian smoothing
def smooth_signal(signal, sigma=2):
    """Apply Gaussian low-pass filter to smooth the signal."""
    return gaussian_filter1d(signal, sigma=sigma, mode="nearest")