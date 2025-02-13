# Train file for MLP and LSTM
import yaml
import torch
#from easydict import EasyDict as edict
import easydict
from utils import update_config
import glob
import json
from sklearn.model_selection import KFold
import numpy as np
from utils import *
from nets import LSTMNet, SimpleMLP

# TODO just for checking
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
        
        # prepocess data based on train set
        X_train, X_val, _, _, _ = preprocess_data(cfg, X_train, X_val, y_train, fold, plotting)
        
        # create Datasets
        print("Creating Datasets...")
        train_dataset = CustomDataset(X_train, y_train, cfg.DATA_PRESET.LABELS)
        val_dataset = CustomDataset(X_val, y_val, cfg.DATA_PRESET.LABELS)
        
        # intializing network
        # default is MLP
        net_type = cfg.TRAIN.get('NET', "mlp")
        input_channels = train_dataset[0][0].shape[1]
        num_layers = 2
        
        if net_type == "lstm":
            net = LSTMNet()
        else:
            net = SimpleMLP()
            
            
            
            
            
            
    
    # crossvalidation
    
    
    ## prepocess data based on train set
    
    ## train Network
    
    ## validate Network
    
    ## save results in tensorboard




if __name__ == '__main__':
    main()
