# Test file for MLP and LSTM
import sys
import os
# Ensure project root is in sys.path
# Need to go two paths up here
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it
    

import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils.preprocess_signals import pad_sequences, normalize_full_signal, normalize_per_timestamp, replace_nan_with_first_value, gaussian_filter1d
from utils.nets import LSTMNet, SimpleMLP
from utils.CustomDataset import CustomDataset
from utils.training_utils import validation, initialize_loss


#TODO Use 1 or many models?
MODEL_PATH = "./pretrained_models/best_model_2025_03_28_14_46_lr0.0001_seed42.pth"
TEST_DATA_PATH = "./data/split_data/test.json"

def main():
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    # Load the checkpoint to the model as we need it for certain data
    print("Loading Model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]

    # Load in the test data
    print("Loading test data...")
    test_data = []
    labels = []
    with open(TEST_DATA_PATH, 'r') as f:
        data_json = json.load(f)
        
        
    for cycle in data_json.values():
        # Extract joint data as (num_joints, time_steps)
        cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in custom_params["choosen_joints"]]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)

        test_data.append(cycle_tensor)
        labels.append(cycle["Label"])

    # Preprocess data based on train data
    print("Preprocessing data...")
    # padding
    test_data = pad_sequences(test_data, max_length=custom_params["train_max_length"], pad_value=float('nan'))
    
    # normalization
    if custom_params["normalization"]:
        print("Normalizing the data...")
        if custom_params["norm_type"] == "full_signal":
            test_data = normalize_full_signal(test_data, custom_params["train_mean"], custom_params["train_std"])
        else:
            test_data = normalize_per_timestamp(test_data, custom_params["train_mean"], custom_params["train_std"])
    
    # adjust padding    
    test_data = replace_nan_with_first_value(test_data)
    
    # smoothing
    if custom_params["smoothing"] > 0:
        test_data = gaussian_filter1d(test_data, sigma=custom_params["smoothing"], axis=2)
        
    # convert data to a tensor
    test_data = torch.tensor(test_data, dtype=torch.float32)

    test_dataset = CustomDataset(test_data, labels, custom_params["labels"])

    test_loader = DataLoader(test_dataset, batch_size=custom_params["batch_size"], shuffle=False)
    
    # Extract input and output sizes
    input_channels = custom_params["input_channels"]
    output_channels = custom_params["output_channels"]

    # initialize the model
    if custom_params["network_type"] == "lstm":
        print("Initializing LSTM...")
        net = LSTMNet(input_channels, 
                    custom_params["hidden_size"], 
                    output_channels, 
                    custom_params["num_layers"], 
                    custom_params["dropout"])

    elif custom_params["network_type"] == "mlp":  # MLP
        print("Initializing MLP...")
        net = SimpleMLP(input_channels, 
                        custom_params["hidden1"], 
                        custom_params["hidden2"], 
                        output_channels)
    else:
        print("Invalid network type, not lstm or mlp")

    # Load weights into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    loss_func = initialize_loss(custom_params, custom_params)

    avg_test_loss, epoch_accuracy, precision, recall, f1, conf_matrix = validation(test_loader, net, loss_func, device, custom_params["network_type"])

    print("Test loss:", avg_test_loss)
    print("Test accuracy:", epoch_accuracy)
    print("Test precision", precision)
    print("Test recall", recall)
    print("Test f1", f1)

    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.show()
    

if __name__ == '__main__':
    main()
