import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles, extract_multivariate_series
from utils.nets import LSTMNet, SimpleMLP
from utils.config import update_config
from utils.split_cycles import split_into_cycles
from utils.preprocess_signals import *
from alphapose.scripts.demo_inference import run_inference

import torch
import numpy as np

# TODO put in different config??
# Type of network that we want to use for the classification
NETWORK_TYPE = "MLP"
# Model path where we want to load the model from
MODEL_PATH = "./pretrained_models/best_model_2025_02_25_15_55_lr0.0001_seed42.pth"
# TODO this is just for test purposes. It is not needed anymore once we get AlphaPose to work, as we do not need to read in the annotated data then
ID = "18_cut"
# INPUT_PATH = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\Annotations\\" + ID + ".json"
# INPUT_VIDEO = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_00" + ID + ".mp4"
INPUT_PATH = os.path.join("E:\SkiProject\AnnotationsByUs", ID[:2] + ".json")
INPUT_VIDEO = r"E:\SkiProject\Cut_videos\DJI_00" + ID + ".mp4"
# path to where all videos are stored
# video_path = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData"
video_path = r"E:\SkiProject\Cut_videos"

def main():
    # TODO put in config file?!
    # Load config 
    
    # TODO Change
    print("Loading config...")
    cfg = update_config("./classification/training/config.yaml") 
    
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    
    # Step 1: Get Keypoints from AlphaPose TODO
    if False:
        print("Loading config...")
        run_args = update_config("./feedback_system/pipe_test.yaml") # TODO Testing set up fix for full pipeline
        output_path, results_list = run_inference(run_args)
        # TODO Convert output to similar to CVAT?

    
    # Step 2: Split into cycles
    input_data = []
    # TODO should not be based on a path
    print("Splitting the data into cycles...")
    cycle_data = split_into_cycles(INPUT_PATH, visualize=False)
    
    # convert to numpy arrays and stack them together
    for cycle in cycle_data.values():

        # Extract joint data as (num_joints, time_steps)
        cycle_data_array = [np.array(cycle[joint], dtype=np.float32) for joint in cfg.DATA_PRESET.CHOOSEN_JOINTS]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data_array)  # Shape: (num_joints, time_steps)

        input_data.append(cycle_tensor)

    # Step 3: Classify user cycles
    # Load the checkpoint to the model
    # TODO modify model path!!
    print("Loading Model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]
    
    # Preprocess data based on train data
    print("Preprocessing data...")
    # padding
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
        # TODO would it be better to get hidden size etc from custom params?! In case we change it in the config file!
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
    for i, cycle_input in enumerate(input_data):
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
        predicted_label = reversed_labels.get(predicted.item())
        print(f"Predicted class: {predicted_label}")
        
        # Step 4: Based on classification use DTW
        
        # based on cycle choose the expert data we want to compare the cycle to
        if predicted_label == "gear3":
            expert_path = "./data/expert_data/expert_cycles_gear3.json"
        elif predicted_label == "gear2":
            # TODO create file!
            print("TODO")
            continue
        else:
            print(f"The system cannot give feedback for {predicted_label}")
            continue
        
        expert_data = load_json(expert_path)
        
        # Define joint triplets for angle comparisons
        # TODO to compare cycles we can either input joint triplets, then we need to set use_keypoints to false
        # otherwise we can input joints, then it will use raw keypoints for DTW
        joint_triplets = [("RHip", "RKnee", "RAnkle"), ("LHip", "LKnee", "LAnkle"), ("RShoulder", "RElbow", "RWrist"), ("LShoulder", "LElbow", "LWrist")]
        joints = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]
        
        # send in data in json format
        cycle = cycle_data[f"Cycle {i+1}"]
        
        dtw_comparisons = compare_selected_cycles(expert_data, cycle, joints, INPUT_VIDEO, video_path, visualize=False)

        # Step 5: Give feedback
        from dtaidistance import dtw
        """
        Plan:
        Have angle from user, start with single signal
        Get best matching expert signal with DTW
            Use frames (time steps) from compare_slected_cycles?
            Use this:
            path = dtw.warping_path(series_user, series_expert, use_ndim=True)
        Plot?
        Filter if error is only one frame/time step
        Convert angles into specific feedback
        Get direction from dtw, in future classify before
        Check if angle from user is off from expert by thershold and show user
        """
        sample_cycle_series, frames_user = extract_multivariate_series(cycle, joint_triplets)

        # TODO JUST FOR PLOTTING REMOVE LATER
        import matplotlib.pyplot as plt
        # Extract the array and the list from the tuple
        data_array, x_values = sample_cycle_series, frames_user

        # Create the plot
        plt.figure(figsize=(10, 6))
        for i in range(data_array.shape[1]):
            plt.plot(x_values, data_array[:, i], label=f'Line {i+1}')

        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Plot of Array Values vs X List')
        plt.legend()
        # Save the plot to a file
        plt.savefig('data/array_vs_list_plot.png')

        closest_cycle = {}
        best_dist = float("inf")
        for _, expert_cycle in expert_data.items():
        
            series_expert, frames_expert = extract_multivariate_series(expert_cycle, joint_triplets)
            print(len(series_expert))
            print(len(sample_cycle_series))
            dist = dtw.distance(sample_cycle_series, series_expert, use_ndim=True)
            if dist < best_dist:
                best_dist = dist
                closest_cycle = expert_cycle
        series_expert, frames_expert = extract_multivariate_series(closest_cycle, joint_triplets)
        path = dtw.warping_path(sample_cycle_series, series_expert, use_ndim=True)
        print(path)



 
    
    

if __name__ == '__main__':
    main()