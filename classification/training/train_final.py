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
from utils.training_utils import training, initialize_loss, initialize_net, set_seed
from utils.config import update_config
from utils.plotting import plot_training_final_metrics

METRICS_NAMES = ["train_losses",
        "train_accs",
        "train_precisions",
        "train_recalls",
        "train_f1s",
        "train_skier_accs"
    ]
    

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
    skier_ids = []
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
        skier_ids.append(cycle["Skier_id"])
    

        
        # Preprocess data based on train data
    print("Preprocessing data...")
    print(train_data[0].shape)
    # padding
    max_length = max(seq.shape[1] for seq in train_data)
    train_data = pad_sequences(train_data, max_length=max_length, pad_value=float('nan'))
    train_data_mean = np.nanmean(train_data, axis=0)
    train_data_std = np.nanstd(train_data, axis=0) 
    
    print(train_data[0].shape)
    # normalization
    if cfg.DATASET.AUG.NORMALIZATION:
        print("Normalizing the data...")
        if cfg.DATASET.AUG.NORM_TYPE == "full_signal":
            train_flattened = train_data.transpose(1, 0, 2).reshape(train_data.shape[1], -1)
            train_data_mean = np.nanmean(train_flattened, axis=1, keepdims=True)  # Shape: (num_joints, 1)
            train_data_std = np.nanstd(train_flattened, axis=1, keepdims=True)    # Shape: (num_joints, 1)
            train_data = normalize_full_signal(train_data, train_data_mean, train_data_std)
        else:
            # Normalize by timestamp
            train_data_mean = np.nanmean(train_data, axis=0)
            train_data_std = np.nanstd(train_data, axis=0) 
            train_data = normalize_per_timestamp(train_data, train_data_mean, train_data_std)
    
    # adjust padding    
    train_data = replace_nan_with_first_value(train_data)
    
    # smoothing
    if cfg.DATASET.AUG.SMOOTHING > 0:
        train_data = gaussian_filter1d(train_data, sigma=cfg.DATASET.AUG.SMOOTHING, axis=2)
        
    # create Datasets
    print("Creating Dataset...")
    train_data = torch.tensor(train_data, dtype=torch.float32)

    train_dataset = CustomDataset(train_data, labels, cfg.DATA_PRESET.LABELS, skier_id=skier_ids)

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
    
    # Get network type to save params
    net_type = cfg.TRAIN.NETWORK.get('NETWORKTYPE', "mlp")
    if net_type.lower() == "mlp":
        custom_params["hidden1"] = cfg.TRAIN.NETWORK.MLP.HIDDEN_1
        custom_params["hidden2"] = cfg.TRAIN.NETWORK.MLP.HIDDEN_2

    else:
        custom_params["hidden_size"] = cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE
        custom_params["num_layers"] = cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS
        custom_params["dropout"] = cfg.TRAIN.NETWORK.LSTM.DROPOUT

    # Get input and output channels
    if cfg.TRAIN.NETWORK.NETWORKTYPE == "lstm":
        input_channels = train_loader.dataset.data.shape[1]
    elif cfg.TRAIN.NETWORK.NETWORKTYPE == "mlp":
        input_channels = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]

    output_channels = len(cfg.DATA_PRESET.LABELS.keys())

    custom_params["input_channels"] = input_channels
    custom_params["output_channels"] = output_channels

    net = initialize_net(cfg, input_channels, output_channels)
    net.to(device)

    # Define loss function and optimizer
    criterion = initialize_loss(cfg)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.LR)  

    set_seed(cfg.TRAIN.SEEDS[0])

    results = {}
    for metric_name in METRICS_NAMES:
        results[metric_name] = []

    # log results to tensorboard
    tensorboard_file_path = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.TENSORBOARD_PATH)
    writer = SummaryWriter(tensorboard_file_path + '/final_train_' +  start_time)

    for epoch in range(cfg.TRAIN.EPOCHS):
        print("Training")
        epoch_train_loss, epoch_train_acc, train_precision, train_recall, train_f1, train_conf_matrix, skier_accs = training(train_loader, net, criterion, optimizer, device, cfg.TRAIN.NETWORK.NETWORKTYPE)
        results["train_losses"].append(epoch_train_loss)
        results["train_accs"].append(epoch_train_acc)
        results["train_precisions"].append(train_precision)
        results["train_recalls"].append(train_recall)
        results["train_f1s"].append(train_f1)
        results["train_skier_accs"].append(skier_accs)


        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar("Recall/Train", train_recall, epoch)
        writer.add_scalar("F1 Score/Train", train_f1, epoch)

        print(f"Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Loss: {epoch_train_loss:.3f}, Accuracy: {epoch_train_acc:.3f}")

    # Save model
    run_dir = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.MODEL_DIR)
    model_dir = os.path.join(run_dir, f"final_train_run_{start_time}_{cfg.TRAIN.NETWORK.NETWORKTYPE}")
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"trained_model_{start_time}_lr{cfg.TRAIN.LR}.pth"
    model_path = os.path.join(model_dir, model_filename)

    # Save the model state dictionary
    #torch.save(net.state_dict(), model_path)
    torch.save({
        "state_dict": net.state_dict(),
        "custom_params" : custom_params
    }, model_path)
    print(f"Saving model {model_path}")
    

    # Save plots
    plot_dir = os.path.join(cfg.LOGGING.ROOT_PATH, cfg.LOGGING.PLOT_PATH)
    plot_dir = os.path.join(plot_dir, "run_final" + start_time)
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_final_metrics(results, plot_dir)

    writer.close()


if __name__ == '__main__':
    main()
