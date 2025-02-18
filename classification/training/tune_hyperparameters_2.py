import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import os
import torch
import json
import glob
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from utils import *
from nets import LSTMNet, SimpleMLP
from torch.utils.tensorboard import SummaryWriter

def train_and_validate(seed, net, criterion, optimizer, cfg, train_loader, val_loader, device, fold = None, start_time = None):
    """
    Trains and validates the model across multiple epochs. Saves the model weights based on lowest validation accuracy. 

    Parameters:
    - seed (int): Random seed for reproducibility.
    - net (torch.nn.Module): Neural network model.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - cfg (Config): Configuration object containing training parameters.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - fold (int): Current cross validation fold. 
    - start_time (str): Start time as string for saving models.

    Returns:
    - results (dict): Dictionary storing loss, accuracy, precision, recall, and F1-score for each epoch.
    - best_train_cm (numpy.ndarray): Best confusion matrix for training.
    - best_val_cm (numpy.ndarray): Best confusion matrix for validation.
    """

    # Set all seeds and make deterministic
    set_seed(seed)

    results = {}
    for metric_name in METRICS_NAMES:
        results[metric_name] = []

    best_val_acc = 0.0
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(cfg.TRAIN.EPOCHS):  # loop over the dataset multiple times
        
        print("Training")
        epoch_train_loss, epoch_train_acc, train_precision, train_recall, train_f1, train_conf_matrix = training(train_loader, net, criterion, optimizer, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["train_losses"].append(epoch_train_loss)
        results["train_accs"].append(epoch_train_acc)
        results["train_precisions"].append(train_precision)
        results["train_recalls"].append(train_recall)
        results["train_f1s"].append(train_f1)

        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_train_loss:.3f}, Accuracy: {epoch_train_acc:.3f}")
        
        print("Validation")
        epoch_val_loss, epoch_val_acc, val_precision, val_recall, val_f1, val_conf_matrix = validation(val_loader, net, criterion, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["val_losses"].append(epoch_val_loss)
        results["val_accs"].append(epoch_val_acc)
        results["val_precisions"].append(val_precision)
        results["val_recalls"].append(val_recall)
        results["val_f1s"].append(val_f1)
        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_val_loss:.3f}, Accuracy: {epoch_val_acc:.3f}")

        # Check early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if not (fold and start_time) == None:
                save_model(net, cfg, fold, seed, start_time)
                print(f"Model saved at epoch {epoch+1}")
            counter = 0  # Reset patience counter
            best_val_cm = val_conf_matrix
            best_train_cm = train_conf_matrix
        else:
            counter += 1  # Increment counter if no loss improvement
        
        if counter >= cfg.TRAIN.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training
        
    print('Finished Training')
    return results, best_train_cm, best_val_cm

def load_dataset(cfg):
    print("Loading Train and validation data...")
    train_val_data = []
    labels = []
    for file in glob.glob(cfg.DATASET.ROOT_PATH + '/*.json'):
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

    return train_val_data, labels

def create_dataloaders(cfg, kf, train_val_data, labels):
    # loop through folds to create dataloaders
    fold_loaders = []
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_data)):
        
        print(f"Fold {fold+1}: Train size = {len(train_index)}, Val size = {len(val_index)}")
        X_train, y_train = ([train_val_data[i] for i in train_index], [labels[i] for i in train_index])
        X_val, y_val = ([train_val_data[i] for i in val_index], [labels[i] for i in val_index])
        
        # prepocess data based on train set
        X_train, X_val, _, _, _ = preprocess_data(cfg, X_train, X_val, y_train, fold)
        
        # create Datasets
        print("Creating Datasets...")
        train_dataset = CustomDataset(X_train, y_train, cfg.DATA_PRESET.LABELS)
        val_dataset = CustomDataset(X_val, y_val, cfg.DATA_PRESET.LABELS)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

        # Store loaders for this fold
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders

def initialize_net(cfg, input_channels, output, hidden_1, hidden_2):
    net_type = cfg.TRAIN.NETWORK.get('NETWORKTYPE', "mlp")
    
    if net_type == "lstm":
        print("Initializing lstm...")
        net = LSTMNet(input_channels, 
                    cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE, 
                    output, 
                    cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS, 
                    cfg.TRAIN.NETWORK.LSTM.DROPOUT)
    else:
        print("Initializing mlp...")
        net = SimpleMLP(input_channels, 
                        hidden_1, 
                        hidden_2, 
                        output)
        
    return net

def train_optimize(config, cfg, data_loaders, output_channels, device):
    """ Training function that Ray Tune will optimize. """
    
    # Extract hyperparameters from Ray Tune
    learning_rate = config["lr"]
    hidden_1 = config["hidden_1"]
    hidden_2 = config["hidden_2"]
    train_loader, val_loader = data_loaders

    if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
        input_channels = train_loader.dataset.data.shape[1]
    elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
        input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
    net = initialize_net(cfg, input_channels, output_channels, hidden_1, hidden_2)
    net.to(device)

    # Define loss function and optimizer
    criterion_type = cfg.TRAIN.get('LOSS', "cross_entropy")
    if criterion_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("Loss type not implemented")

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

    results, _, _ = train_and_validate(cfg.TRAIN.SEEDS[0], net, criterion, optimizer, cfg, train_loader, val_loader, device)
    
    best_val_acc = max(results["val_accs"])
    best_val_loss = min(results["val_losses"])
    # Log metric for Ray Tune optimization
    ray.train.report(dict(accuracy=best_val_acc, loss=best_val_loss))



def tune_hyperparameters():
    """Runs Ray Tune to optimize hyperparameters."""

    # Define search space for hyperparameters
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),  # Learning rate search space
        #"batch_size": tune.choice([16, 32, 64, 128]),  # Different batch sizes
        #"hidden_units": tune.randint(32, 256)  # Number of hidden units
        "hidden_1" : tune.randint(32, 256),
        "hidden_2" : tune.randint(32, 256)
    }

    # Load config
    cfg = update_config("config.yaml")
    
    # Select device (CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load dataset
    train_val_data, labels = load_dataset(cfg)
    
    # K-Fold Cross-validation setup
    kf = KFold(n_splits=cfg.TRAIN.K_FOLDS, shuffle=True, random_state=cfg.TRAIN.SEEDS[0])
    fold_loaders = create_dataloaders(cfg, kf, train_val_data, labels)
    
    output_channels = len(set(labels))  # Number of output classes

    # Use HyperOpt for efficient searching
    search_algo = HyperOptSearch(metric="loss", mode="min")
    
    # Scheduler for early stopping
    scheduler = ASHAScheduler(
        # metric="accuracy", 
        # mode="max", 
        grace_period=5, 
        reduction_factor=1.5)

    # Run hyperparameter optimization with Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_optimize, cfg=cfg, data_loaders=fold_loaders[0], output_channels=output_channels, device=device),
        config=search_space,
        num_samples=10,  # Number of trials
        search_alg=search_algo,
        scheduler=scheduler,
        metric="loss",
        mode="min"
    )

    print("Best hyperparameters found:", analysis.best_config)

if __name__ == '__main__':
    tune_hyperparameters()