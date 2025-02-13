# Train file for MLP and LSTM
import yaml
import torch
#from easydict import EasyDict as edict
import easydict
from utils import update_config
import glob
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

plotting = True

def main():
    # Load config 
    print("Loading config...")
    cfg = update_config("config.yaml")
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")

    # TODO do not need file_name??!
    file_name = cfg.DATASET.FILE_PREFIX # Prefix for json files with annotated keypoints
    path = cfg.DATASET.ROOT_PATH # Path to folder with annotated jsons
    
    # load data
    print("Loading Train and validation data...")
    train_val_data = []
    labels = []
    for file in glob.glob(path + '/*.json'):
            # TODO make json load function
            with open(file, 'r') as f:
                data_json = json.load(f)
                
            for cycle in data_json.values():
                # TODO not sure if transforming to tensors should be done here already
                # Extract joint data as (num_joints, time_steps)
                cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in cfg.DATA_PRESET.CHOOSEN_JOINTS]

                # Stack into a (num_joints, time_steps) tensor
                cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)
                #longest_cycle = max(longest_cycle, cycle_tensor.shape[1])  # Update max length

                train_val_data.append(cycle_tensor)
                labels.append(cycle["Label"])
    
    # create train and val dataloaders for crossvalidation
    
    ## Initialize KFold (5 splits)
    # TODO seed!
    # TODO check test size in config!
    kf = KFold(n_splits=cfg.TRAIN.K_FOLDS, shuffle=True, random_state=42)
    # loop through folds to create dataloaders
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_data)):
        
        print(f"Fold {fold+1}: Train size = {len(train_index)}, Val size = {len(val_index)}")
        X_train, y_train = ([train_val_data[i] for i in train_index], [labels[i] for i in train_index])
        X_val, y_val = ([train_val_data[i] for i in val_index], [labels[i] for i in val_index])

        ## prepocess data based on train set
        # Pad the sequences to have the same length in both X_train and X_val
        max_length = max(seq.shape[1] for seq in X_train)  # Find the max length in X_train
        X_train = pad_sequences(X_train, max_length=max_length, pad_value=float('nan'))
        X_val = pad_sequences(X_val, max_length=max_length, pad_value=float('nan'))
        
        # Normalize the padded training data
        if cfg.DATASET.AUG.NORMALIZATION:
            # TODO just for sanity checking
            X_train_raw = X_train
            print("Normalizing the data...")
            
            norm_type = cfg.DATASET.AUG.get('NORM_TYPE', "full_signal")
            if norm_type == "per_timestamp":
                print("Normalizing the signal per timestamp")
                
                mean = np.nanmean(X_train, axis=0)  # Shape: (num_joints, time_steps)
                std = np.nanstd(X_train, axis=0) 
                
                # normalize the train and val data
                X_train = normalize_per_timestamp(X_train, mean, std)
                X_val = normalize_per_timestamp(X_val, mean, std)
            
            else:
                print("Normalizing the full signal")
                
                # Reshape to merge time steps: (num_joints, num_samples * time_steps)
                X_train_flattened = X_train.transpose(1, 0, 2).reshape(X_train.shape[1], -1)

                # Compute mean and std along the flattened axis
                mean = np.nanmean(X_train_flattened, axis=1, keepdims=True)  # Shape: (num_joints, 1)
                std = np.nanstd(X_train_flattened, axis=1, keepdims=True)    # Shape: (num_joints, 1)
                
                # normalize the train and val data
                X_train = normalize_full_signal(X_train, mean, std)
                X_val = normalize_full_signal(X_val, mean, std)
                
            if plotting:
                plot_raw_vs_normalized(X_train_raw, X_train, y_train)
            
            
            
    
    # crossvalidation
    
    
    ## prepocess data based on train set
    
    ## train Network
    
    ## validate Network
    
    ## save results in tensorboard




if __name__ == '__main__':
    main()
