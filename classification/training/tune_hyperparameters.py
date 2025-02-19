import ray
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import os
import torch
import json
import glob
import tempfile
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils import *
from nets import LSTMNet, SimpleMLP
from torch.utils.tensorboard import SummaryWriter

# TODO OPTIMIZE
# Scheduler
# Not reproducible anymore

def load_dataset(cfg):
    train_val_data = []
    labels = []
    for file in glob.glob(cfg.DATASET.ROOT_ABSOLUTE_PATH + '/*.json'):
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

def create_dataloaders(config, cfg, train_val_data, labels):
    # Split the data 
    train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, 
                                                                      labels, 
                                                                      test_size=cfg.DATASET.VAL_SIZE,
                                                                      train_size=cfg.DATASET.TRAIN_SIZE,
                                                                      random_state=cfg.TRAIN.SEEDS[0])
    
    # prepocess data based on train set
    train_data, val_data, _, _, _ = preprocess_data(cfg, train_data, val_data, train_labels)
    
    # create Datasets
    print("Creating Datasets...")
    train_dataset = CustomDataset(train_data, train_labels, cfg.DATA_PRESET.LABELS)
    val_dataset = CustomDataset(val_data, val_labels, cfg.DATA_PRESET.LABELS)
    # create dataloaders

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    return train_loader, val_loader

def initialize_net(cfg, input_channels, output, config):
    net_type = cfg.TRAIN.NETWORK.get('NETWORKTYPE', "mlp")
    
    if net_type == "lstm":
        print("Initializing lstm...")
        net = LSTMNet(input_channels, 
                    config["hidden_size"], 
                    output, 
                    config["num_layers"], 
                    config["dropout"])
    else:
        print("Initializing mlp...")
        net = SimpleMLP(input_channels, 
                        config["hidden_1"], 
                        config["hidden_2"], 
                        output)
        
    return net

def train_func(config, cfg):
    # Select device (CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data and put into dataloaders
    train_val_data, labels = load_dataset(cfg)

    train_loader, val_loader = create_dataloaders(config, cfg, train_val_data, labels)

    output_channels = len(set(labels))

    if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
        input_channels = train_loader.dataset.data.shape[1]
    elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
        input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]
    net = initialize_net(cfg, input_channels, output_channels, config)
    net.to(device)

    if cfg.OPTIMIZATION.CHECKPOINTS.ENABLE:
        # Load existing checkpoint through `get_checkpoint()` API.
        if train.get_checkpoint():
            loaded_checkpoint = train.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                )
                net.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

    # Define loss function and optimizer
    criterion = initialize_loss(cfg, config)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])  

    for epoch in range(cfg.TRAIN.EPOCHS):
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
                inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                if cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
                    inputs = inputs.view(inputs.size(0), -1)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        if cfg.OPTIMIZATION.CHECKPOINTS.ENABLE == True:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (net.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": (val_loss / val_steps), "accuracy": correct / total},
                    checkpoint=checkpoint,
                )
        else:
            train.report({"loss": (val_loss / val_steps), "accuracy": correct / total})

    print("Finished Training")

def set_ray_seed(seed):
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONHASHSEED": str(seed)}})

def tune_hyperparameters():
    # Load config
    cfg = update_config("config.yaml")
    # Load search space params
    ssp = update_config(cfg.OPTIMIZATION.SEARCH_CONFIG)

    
    for seed in ssp.SEEDS:
        set_seed(seed)
        set_ray_seed(seed)

        if cfg.TRAIN.NETWORK.NETWORKTYPE != ssp.NETWORK.NETWORKTYPE:
            raise ValueError(f"Error: Different config ({cfg.TRAIN.NETWORK.NETWORKTYPE})"
                            f" and serach space network ({ssp.NETWORK.NETWORKTYPE}) type!")
        if cfg.TRAIN.EPOCHS != ssp.MAX_EPOCHS:
            print("#"*100)
            print(f"***WARNING***")
            print(f"Not the same amount of epochs in config and search space, search might end sooner than given MAX_EPOCHS")
            print("#"*100)

        if  ssp.NETWORK.NETWORKTYPE == "mlp":
            # Define search space for hyperparameters
            search_space = {
                "lr": tune.loguniform(ssp.LR.MIN, ssp.LR.MAX),  # Learning rate search space
                "batch_size": tune.choice(ssp.BATCH_SIZE),  # Different batch sizes
                "hidden_1" : tune.randint(ssp.NETWORK.MLP.HIDDEN_MIN_SIZE, ssp.NETWORK.MLP.HIDDEN_MAX_SIZE),
                "hidden_2" : tune.randint(ssp.NETWORK.MLP.HIDDEN_MIN_SIZE, ssp.NETWORK.MLP.HIDDEN_MAX_SIZE),
                "loss_type": tune.choice(ssp.LOSS_TYPE)
            }
        elif ssp.NETWORK.NETWORKTYPE == "lstm":
            search_space = {
                "lr": tune.loguniform(ssp.LR.MIN, ssp.LR.MAX),  # Learning rate search space
                "batch_size": tune.choice(ssp.BATCH_SIZE),  # Different batch sizes
                "hidden_size": tune.randint(ssp.NETWORK.LSTM.HIDDEN_MIN_SIZE, ssp.NETWORK.LSTM.HIDDEN_MAX_SIZE),  # Number of hidden units
                "num_layers": tune.choice(ssp.NETWORK.LSTM.NUM_LAYERS),
                "dropout": tune.choice(ssp.NETWORK.LSTM.DROPOUT)
            }


        scheduler = ASHAScheduler(
            max_t=ssp.MAX_EPOCHS,
            grace_period=ssp.GRACE_PERIOD,
            reduction_factor=ssp.REDUCTION_FACTOR)
        
        run_folder_name = "raytune_experiment_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_func, cfg=cfg),
                resources={"cpu": 4, "gpu": 0}
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=ssp.NUM_SAMPLES,
            ),
            run_config=RunConfig(
            storage_path=cfg.OPTIMIZATION.OUTPUT_ROOT,  # Change this to any folder
            name=run_folder_name
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        
        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]))
        
        # Save best results to a json
        best_trial_results = {
            "best_config": best_result.config,
            "final_validation_loss": best_result.metrics["loss"],
            "final_validation_accuracy": best_result.metrics["accuracy"],
            "seed": seed
        }
        json_path = os.path.join(cfg.OPTIMIZATION.OUTPUT_ROOT, run_folder_name)
        json_path = os.path.join(json_path, "raytune_results.json")
        with open(json_path, "w") as f:
            json.dump(best_trial_results, f, indent=4)

        if not ssp.MULTIPLE_SEEDS:
            break
    
if __name__ == '__main__':
    tune_hyperparameters()