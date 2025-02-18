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

def train_model(config, cfg, fold_loaders, output_channels, device):
    """ Training function that Ray Tune will optimize. """
    
    # Extract hyperparameters from Ray Tune
    learning_rate = config["lr"]
    # batch_size = config["batch_size"]
    # hidden_units = config["hidden_units"]
    
    # Load training data for cross-validation
    all_results, best_train_cms, best_val_cms = cross_validation(
        cfg, fold_loaders, output_channels, device, lr=learning_rate)

    # Calculate average performance
    average_results = calc_avg_metrics(cfg.TRAIN.K_FOLDS, all_results, cfg.TRAIN.SEEDS, cfg.TRAIN.EPOCHS)
    # print("\n\n\nKeys in average_results:", average_results.keys())  # Debugging line
    # print(average_results)

    avg_metrics = {
    "accuracy": [],
    "loss": [],
    "precision": [],
    "recall": [],
    "f1": []
    }

    for fold in average_results.keys():
        fold_data = all_results[fold]  # Get metrics for current fold
        print(fold_data)

        avg_metrics["accuracy"].append(fold_data["results"]["val_accs"][-1])  # Last epoch accuracy
        avg_metrics["loss"].append(fold_data["results"]["val_losses"][-1])
        avg_metrics["precision"].append(fold_data["results"]["val_precisions"][-1])
        avg_metrics["recall"].append(fold_data["results"]["val_recalls"][-1])
        avg_metrics["f1"].append(fold_data["results"]["val_f1s"][-1])

    # Compute mean across all folds
    final_results = {key: np.mean(val) for key, val in avg_metrics.items()}

    # Log metric for Ray Tune optimization
    tune.report(accuracy=final_results["accuracy"], loss=final_results["loss"])

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

def cross_validation(cfg, fold_loaders, output_channels, device, lr=None):
    """
    Performs k-fold cross-validation on the dataset. Creates a new net for each seed in each fold. 

    Parameters:
    - cfg (Config): Configuration object containing training settings.
    - fold_loaders (list): List of (train_loader, val_loader) tuples for each fold.
    - output_channels (int): Number of output classes.
    - device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
    - all_results (list): List of dictionaries storing results for each fold and seed.
    - best_train_cms (list): List of best training confusion matrices.
    - best_val_cms (list): List of best validation confusion matrices.
    """

    all_results = []
    best_train_cms = []
    best_val_cms = []

    # Log start time for model saving
    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n>>> Training on Fold {fold+1} <<<\n")
        if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
            input_channels = train_loader.dataset.data.shape[1]
        elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
            input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
        
        # Initialize a new model for each fold
        for seed in cfg.TRAIN.SEEDS:
            print(f"\n========== Running for Seed {seed} on Fold {fold+1} ==========\n")
            
            # Set seed for reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Create neural network
            net = initialize_net(cfg, input_channels, output_channels)
            net.to(device)

            # Define loss function and optimizer
            criterion_type = cfg.TRAIN.get('LOSS', "cross_entropy")
            if criterion_type == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()
            else:
                print("Loss type not implemented")
            
            # Check if lr should come from cfg or raytune optimization
            if lr == None:
                optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.LR)
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            # Train and validate
            results, best_train_cm, best_val_cm = train_and_validate(seed, 
                                                                    net, 
                                                                    criterion, 
                                                                    optimizer,
                                                                    cfg,
                                                                    train_loader,
                                                                    val_loader,
                                                                    device,
                                                                    fold,
                                                                    start_time
                                                                    )
            
            # Store results
            all_results.append({'seed': seed, 'fold': fold+1, 'results': results})
            best_train_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_train_cm})
            best_val_cms.append({'seed': seed, 'fold': fold+1, 'cm': best_val_cm})
    return all_results, best_train_cms, best_val_cms


def tune_hyperparameters():
    """Runs Ray Tune to optimize hyperparameters."""

    # Define search space for hyperparameters
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),  # Learning rate search space
        #"batch_size": tune.choice([16, 32, 64, 128]),  # Different batch sizes
        #"hidden_units": tune.randint(32, 256)  # Number of hidden units
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
    search_algo = HyperOptSearch(metric="accuracy", mode="max")
    
    # Scheduler for early stopping
    scheduler = ASHAScheduler(metric="accuracy", mode="max", grace_period=5, reduction_factor=2)

    # Run hyperparameter optimization with Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_model, cfg=cfg, fold_loaders=fold_loaders, output_channels=output_channels, device=device),
        config=search_space,
        num_samples=3,  # Number of trials
        search_alg=search_algo,
        scheduler=scheduler
    )

    print("Best hyperparameters found:", analysis.best_config)


if __name__ == '__main__':
    tune_hyperparameters()
