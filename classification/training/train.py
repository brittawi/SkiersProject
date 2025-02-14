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
from torch.utils.tensorboard import SummaryWriter

# TODO just for checking
plotting = False

def main():
    # Load config 
    print("Loading config...")
    cfg = update_config("config.yaml")
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")

    # # TODO do not need file_name??!
    # file_name = cfg.DATASET.FILE_PREFIX # Prefix for json files with annotated keypoints
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
            # Extract joint data as (num_joints, time_steps)
            cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in cfg.DATA_PRESET.CHOOSEN_JOINTS]

            # Stack into a (num_joints, time_steps) tensor
            cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)

            train_val_data.append(cycle_tensor)
            labels.append(cycle["Label"])
    
    # create train and val dataloaders for crossvalidation
    
    ## Initialize KFold (5 splits)
    # TODO seed! -> Use first in train list of seeds?(Added)
    # TODO check test size in config!
    kf = KFold(n_splits=cfg.TRAIN.K_FOLDS, shuffle=True, random_state=cfg.TRAIN.SEEDS[0])
    # loop through folds to create dataloaders
    fold_loaders = []
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
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

        # Store loaders for this fold
        fold_loaders.append((train_loader, val_loader))
    

    output_channels = len(set(train_dataset.labels))
    
    # intializing network
    # TODO Moved to utils
    # default is MLP
# def initialize_net(cfg, input_channels, output):
#     net_type = cfg.TRAIN.get('NET', "mlp")
    
#     if net_type == "lstm":
#         print("Initializing lstm...")
#         net = LSTMNet(input_channels, 
#                     cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE, 
#                     output, 
#                     cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS, 
#                     cfg.TRAIN.NETWORK.LSTM.DROPOUT)
#     else:
#         print("Initializing mlp...")
#         net = SimpleMLP(input_channels, 
#                         cfg.TRAIN.NETWORK.MLP.HIDDEN_1, 
#                         cfg.TRAIN.NETWORK.MLP.HIDDEN_2, 
#                         output)
        
#     return net
     
    # training the network
    all_results, best_train_cms, best_val_cms = cross_validation(cfg, fold_loaders, output_channels, device)

    # log results to tensorboard
    tensorboard_file_path = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.TENSORBOARD_PATH)
    writer = SummaryWriter(tensorboard_file_path + '/cross_validation_experiment_' + datetime.now().strftime("%Y_%m_%d_%H_%M") ) # Format as HH:MM:SS)
    average_results = calc_avg_metrics(cfg.TRAIN.K_FOLDS, all_results, cfg.TRAIN.SEEDS, cfg.TRAIN.EPOCHS)
    write_cv_results(average_results, cfg.TRAIN.EPOCHS, writer)     
    writer.close()


        
        
            
            
            
            
            
            
    
    # crossvalidation
    
    
    ## prepocess data based on train set
    
    ## train Network
    
    ## validate Network
    
    ## save results in tensorboard




if __name__ == '__main__':
    main()
