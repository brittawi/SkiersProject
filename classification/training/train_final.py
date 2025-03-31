# Train file for MLP and LSTM
import sys
import os

# Ensure project root is in sys.path
# Need to go two paths up here
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it
    
import torch
from torch.utils.data import DataLoader
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from utils.preprocess_signals import pad_sequences, normalize_full_signal, normalize_per_timestamp, replace_nan_with_first_value, gaussian_filter1d
from utils.CustomDataset import CustomDataset
from utils.training_utils import calc_avg_metrics
from utils.config import update_config

# TODO just for checking
plotting = False

def main():
    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    print(f"Starting run {start_time}...")
    
    # Load config 
    print("Loading config...")
    cfg = update_config("./classification/training/config.yaml")
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    
    # load data
    print("Loading Train data...")
    train_data = []
    labels = []
    file = os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.TRAIN_FILE_NAME)
    with open(file, 'r') as f:
        data_json = json.load(f)
        
        
    for cycle in data_json.values():
        # Extract joint data as (num_joints, time_steps)
        cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in cfg.DATA_PRESET.CHOOSEN_JOINTS]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)

        train_data.append(cycle_tensor)
        labels.append(cycle["Label"])
    

        
        # Preprocess data based on train data
    print("Preprocessing data...")
    # padding
    max_length = max(seq.shape[1] for seq in train_data)
    train_data_mean = np.nanmean(train_data, axis=0)
    train_data_std = np.nanstd(train_data, axis=0) 
    train_data = pad_sequences(train_data, max_length=max_length, pad_value=float('nan'))
    
    # normalization
    if cfg.DATASET.AUG.NORMALIZATION:
        print("Normalizing the data...")
        if cfg.DATASET.AUG.NORM_TYPE == "full_signal":
            train_data = normalize_full_signal(train_data, train_data_mean, train_data_std)
        else:
            train_data = normalize_per_timestamp(train_data, train_data_mean, train_data_std)
    
    # adjust padding    
    train_data = replace_nan_with_first_value(train_data)
    
    # smoothing
    if cfg.DATASET.AUG.SMOOTHING > 0:
        train_data = gaussian_filter1d(train_data, sigma=cfg.DATASET.AUG.SMOOTHING, axis=2)
        
    # convert data to a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    train_dataset = CustomDataset(train_data, labels, cfg.DATA_PRESET.LABELS)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    

    # params that need to be saved with the statedict for later use
    custom_params = {
        "train_mean" : train_data_mean,
        "train_std" : train_data_std,
        "train_max_length" : max_length,
        "normalization" : cfg.DATASET.AUG.NORMALIZATION,
        "norm_type" : cfg.DATASET.AUG.get('NORM_TYPE', "full_signal"),
        "smoothing" : cfg.DATASET.AUG.SMOOTHING,
        "choosen_joints" : cfg.DATA_PRESET.CHOOSEN_JOINTS,
        "labels" : cfg.DATA_PRESET.LABELS,
        "network_type" : cfg.TRAIN.NETWORK.NETWORKTYPE,
        "batch_size" : cfg.TRAIN.BATCH_SIZE,
        "loss_type" : cfg.TRAIN.LOSS
    }
    
    net_type = cfg.TRAIN.NETWORK.get('NETWORKTYPE', "mlp")
    if net_type.lower() == "mlp":
        custom_params["hidden1"] = cfg.TRAIN.NETWORK.MLP.HIDDEN_1
        custom_params["hidden2"] = cfg.TRAIN.NETWORK.MLP.HIDDEN_2
    else:
        custom_params["hidden_size"] = cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE
        custom_params["num_layers"] = cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS
        custom_params["dropout"] = cfg.TRAIN.NETWORK.LSTM.DROPOUT
        
    
    # create Datasets
    print("Creating Datasets...")
    train_dataset = CustomDataset(X_train, y_train, cfg.DATA_PRESET.LABELS)
    val_dataset = CustomDataset(X_val, y_val, cfg.DATA_PRESET.LABELS)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    # Store loaders for this fold and custom params which contain the mean and std for the train set
    fold_loaders.append((train_loader, val_loader, custom_params))
    

    output_channels = len(cfg.DATA_PRESET.LABELS.keys())


    # log results to tensorboard
    tensorboard_file_path = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.TENSORBOARD_PATH)
    writer = SummaryWriter(tensorboard_file_path + '/cross_validation_experiment_' +  start_time) 
    writer.close()


if __name__ == '__main__':
    main()
