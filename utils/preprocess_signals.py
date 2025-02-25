import numpy as np
from scipy.ndimage import gaussian_filter1d
from utils.plotting import plot_raw_vs_normalized
    
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

def normalize_per_timestamp(train_val_data, mean, std):
    """
    Normalizes data per timestamp and per joint.
    
    Parameters:
    - train_val_data: list of NumPy arrays of shape (num_joints, time_steps)

    Returns:
    - Normalized list of NumPy arrays with the same shape
    """
    
    # Normalize (avoid division by zero with epsilon)
    normalized_data = (train_val_data - mean) / (std + 1e-8)
    
    return normalized_data
    
def normalize_full_signal(train_val_data, mean, std):
    """
    Normalizes data per joint over the full signal (all timestamps concatenated).
    
    Parameters:
    - train_val_data: list of NumPy arrays of shape (num_joints, time_steps)

    Returns:
    - Normalized list of NumPy arrays with the same shape
    """

    # # Reshape to merge time steps: (num_joints, num_samples * time_steps)
    flattened = train_val_data.transpose(1, 0, 2).reshape(train_val_data.shape[1], -1)

    # Normalize each joint across all time steps
    normalized_flattened = (flattened - mean) / (std + 1e-8)

    # Reshape back to (num_samples, num_joints, time_steps)
    normalized_data = normalized_flattened.reshape(train_val_data.shape[1], train_val_data.shape[0], train_val_data.shape[2])
    normalized_data = normalized_data.transpose(1, 0, 2)  # Back to (num_samples, num_joints, time_steps)
    
    return normalized_data
    
def pad_sequences(sequences, max_length=None, pad_value=0.0):
    """
    Pads each sequence in the list to the specified max_length with a custom value.
    
    Parameters:
    - sequences: list of NumPy arrays (each of shape (num_joints, time_steps))
    - max_length: the desired length of the time series dimension (default None, will use the max length of sequences in sequences)
    - pad_value: the value to use for padding (default is 0.0)

    Returns:
    - List of padded NumPy arrays with shape (num_joints, max_length)
    """
    # Determine max_length if not provided
    if max_length is None:
        max_length = max(seq.shape[1] for seq in sequences)
    
    padded_sequences = []
    
    for seq in sequences:
        pad_length = max_length - seq.shape[1]
        
        # Pad if needed
        if pad_length > 0:
            padded_seq = np.pad(seq, ((0, 0), (0, pad_length)), mode='constant', constant_values=pad_value)
        # cut sequence if needed
        elif pad_length < 0:
            padded_seq = seq[:, :max_length]
        else:
            padded_seq = seq  # If already max_length, keep as is
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

def replace_nan_with_first_value(arr):
    """
    Replaces NaN values in each sequence with the first non-NaN value.
    
    Parameters:
        arr (numpy.ndarray): Input array of shape (samples, joints, time_steps).
    
    Returns:
        numpy.ndarray: Array with NaNs replaced.
    """
    mask = np.isnan(arr)  # Find NaN positions
    for sample in range(arr.shape[0]):  # Iterate over samples
        for joint in range(arr.shape[1]):  # Iterate over joints
            # Find the first non-NaN value
            valid_values = arr[sample, joint, ~mask[sample, joint]]
            if valid_values.size > 0:  # If there are valid values
                # TODO 
                first_value = valid_values[-1]  # Take the first valid number
                arr[sample, joint, mask[sample, joint]] = first_value  # Replace NaNs with it

    return arr
               
def preprocess_data(cfg, X_train, X_val, y_train, fold = 1, plotting=False):
    # Pad the sequences to have the same length in both X_train and X_val
    max_length = max(seq.shape[1] for seq in X_train)  # Find the max length in X_train
    X_train = pad_sequences(X_train, max_length=max_length, pad_value=float('nan'))
    X_val = pad_sequences(X_val, max_length=max_length, pad_value=float('nan'))
    
    # TODO just for sanity checking
    X_train_raw = X_train
        
    # Normalize the padded training data
    if cfg.DATASET.AUG.NORMALIZATION:
        print("Normalizing the data...")
        
        norm_type = cfg.DATASET.AUG.get('NORM_TYPE', "full_signal")
        if norm_type == "per_timestamp":
            print("Normalizing the signal per timestamp...")
            
            # compute mean and std per timestamp
            mean = np.nanmean(X_train, axis=0)  # Shape: (num_joints, time_steps)
            std = np.nanstd(X_train, axis=0) 
            
            # normalize the train and val data
            X_train = normalize_per_timestamp(X_train, mean, std)
            X_val = normalize_per_timestamp(X_val, mean, std)
        
        else:
            print("Normalizing the full signal...")
            
            # Reshape to merge time steps: (num_joints, num_samples * time_steps)
            X_train_flattened = X_train.transpose(1, 0, 2).reshape(X_train.shape[1], -1)

            # Compute mean and std along the flattened axis
            mean = np.nanmean(X_train_flattened, axis=1, keepdims=True)  # Shape: (num_joints, 1)
            std = np.nanstd(X_train_flattened, axis=1, keepdims=True)    # Shape: (num_joints, 1)
            
            # normalize the train and val data
            X_train = normalize_full_signal(X_train, mean, std)
            X_val = normalize_full_signal(X_val, mean, std)
    
    # TODO when to replace padding?!!
    # replace padded values with start of sequence
    X_train = replace_nan_with_first_value(X_train)
    X_val = replace_nan_with_first_value(X_val)
    
    # Smooth the signal
    if cfg.DATASET.AUG.SMOOTHING > 0:
        print("Smoothing the signal")
        X_train = gaussian_filter1d(X_train, sigma=cfg.DATASET.AUG.SMOOTHING, axis=2)
        
        
    if plotting and fold == 0:
        plot_raw_vs_normalized(X_train_raw, X_train, y_train)
        
    return X_train, X_val, mean, std, max_length