import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles
from utils.nets import LSTMNet, SimpleMLP
from utils.config import update_config
from utils.split_cycles import split_into_cycles
from utils.preprocess_signals import *

import torch
import numpy as np

# TODO put in different config??
NETWORK_TYPE = "MLP"
MODEL_PATH = "./classification/training/runs/saved_models/run_2025_02_25_15_55_mlp/fold_1/best_model_2025_02_25_15_55_lr0.0001_seed42.pth"
# TODO this is just for test purposes. It is not needed anymore once we get AlphaPose to work, as we do not need to read in the annotated data then
INPUT_PATH = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\Annotations\38.json"

def main():
    # TODO put in config file?!
    # Load config 
    print("Loading config...")
    cfg = update_config("./classification/training/config.yaml") 
    
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    
    # Step 1: Get Keypoints from AlphaPose
    
    # Step 2: Split into cycles
    input_data = []
    # TODO should not be based on a path
    cycle_data = split_into_cycles(INPUT_PATH, visualize=False)
    
    # convert to numpy arrays and stack them together
    for cycle in cycle_data.values():

        # Extract joint data as (num_joints, time_steps)
        cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in cfg.DATA_PRESET.CHOOSEN_JOINTS]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)

        input_data.append(cycle_tensor)

    # Step 3: Classify user cycles
    # Load the checkpoint to the model
    # TODO modify model path!!
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]
    
    # Preprocess data based on train data
    # padding
    print("max length", custom_params["train_max_length"])
    input_data = pad_sequences(input_data, max_length=custom_params["train_max_length"], pad_value=float('nan'))
    
    # normalization
    if custom_params["normalization"]:
        print("Normalizing the data...")
        if custom_params["norm_type"] == "full_signal":
            input_data = normalize_full_signal(input_data, custom_params["train_mean"], custom_params["train_std"])
        else:
            input_data = normalize_per_timestamp(input_data, custom_params["train_mean"], custom_params["train_std"])
    
    # adjust padding    
    input_data = replace_nan_with_first_value(input_data)
    
    # smoothing
    if custom_params["smoothing"] > 0:
        input_data = gaussian_filter1d(input_data, sigma=custom_params["smoothing"], axis=2)
        
    # convert data to a tensor
    input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Extract input and output sizes
    input_channels = custom_params["input_channels"]
    output_channels = custom_params["output_channels"]

    # initialize the model
    if NETWORK_TYPE == "LSTM":

        print("Initializing LSTM...")
        net = LSTMNet(input_channels, 
                    cfg.TRAIN.NETWORK.LSTM.HIDDEN_SIZE, 
                    output_channels, 
                    cfg.TRAIN.NETWORK.LSTM.NUM_LAYERS, 
                    cfg.TRAIN.NETWORK.LSTM.DROPOUT)

    else:  # MLP

        print("Initializing MLP...")
        net = SimpleMLP(input_channels, 
                        cfg.TRAIN.NETWORK.MLP.HIDDEN_1, 
                        cfg.TRAIN.NETWORK.MLP.HIDDEN_2, 
                        output_channels)

    # Load weights into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    print(f"Loaded model with input={input_channels}, output={output_channels}")
    
    # reverse label dict to get predictions
    reversed_labels = {v: k for k, v in cfg.DATA_PRESET.LABELS.items()}
    
    # classify each cycle
    for cycle_input in input_data:
        # this is done in the dataloader 
        cycle_input = cycle_input.T # (12, 97) => (97, 12), 12 joints, 97 timesteps
        # add batch size
        cycle_input = cycle_input.unsqueeze(0)
        
        if NETWORK_TYPE == "MLP":
            cycle_input = cycle_input.contiguous().view(cycle_input.size(0), -1)
        cycle_input = cycle_input.to(device)
        outputs = net(cycle_input)
        outputs = outputs.squeeze(1)
            
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        print(f"Predicted class: {reversed_labels.get(predicted.item())}")

    # Step 4: Based on classification use DTW
    
    # Load cycle data
    #id = "15_cut" # r->l gear 3, cycle 5
    id = "22_cut" # l->r gear 3, cycle 5
    #id = "38" # front gear 3, cycle 5
    file_path = "./data/labeled_data/labeled_cycles_" + id + ".json"
    cycle_to_compare = "Cycle 5"
    user_video = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_00" + id + ".mp4"  # Path to the corresponding video file
    data = load_json(file_path)
    user_data = data.get(cycle_to_compare)
    
    # Load expert data
    expert_path = "./data/expert_data/expert_cycles_gear3.json"
    expert_data = load_json(expert_path)
    
    video_path = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData"

    # Define joint triplets for angle comparisons
    # TODO to compare cycles we can either input joint triplets, then we need to set use_keypoints to false
    # otherwise we can input joints, then it will use raw keypoints for DTW
    joint_triplets = [("RHip", "RKnee", "RAnkle"), ("LHip", "LKnee", "LAnkle"), ("RShoulder", "RElbow", "RWrist"), ("LShoulder", "LElbow", "LWrist")]
    joints = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]
    
    dtw_comparisons = compare_selected_cycles(expert_data, user_data, joints, user_video, video_path, visualize=False)
    

if __name__ == '__main__':
    main()