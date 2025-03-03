import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles, extract_frame
from utils.feedback_utils import extract_multivariate_series_for_lines, calculate_differences, draw_lines_and_text
from utils.nets import LSTMNet, SimpleMLP
from utils.config import update_config
from utils.split_cycles import split_into_cycles
from utils.preprocess_signals import *
from utils.annotation_format import halpe26_to_coco
from utils.plotting import plot_lines
from alphapose.scripts.demo_inference import run_inference

import torch
import numpy as np
import cv2

# # TODO put in different config??
# # Type of network that we want to use for the classification
# NETWORK_TYPE = "MLP"
# # Model path where we want to load the model from
# MODEL_PATH = "./pretrained_models/best_model_2025_02_25_15_55_lr0.0001_seed42.pth"
# # TODO this is just for test purposes. It is not needed anymore once we get AlphaPose to work, as we do not need to read in the annotated data then
ID = "38"
# # INPUT_PATH = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\Annotations\\" + ID + ".json"
# # INPUT_VIDEO = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_00" + ID + ".mp4"
INPUT_PATH = os.path.join("E:\SkiProject\AnnotationsByUs", ID[:2] + ".json")
# INPUT_VIDEO = r"E:\SkiProject\Cut_videos\DJI_00" + ID + ".mp4"
# # path to where all videos are stored
# # video_path = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData"
# video_path = r"E:\SkiProject\Cut_videos"
testing_with_inference = False

def main():
    
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    
    # Step 1: Get Keypoints from AlphaPose 
    
    print("Loading config...")
    run_args = update_config("./feedback_system/pipe_test.yaml") # TODO Testing set up fix for full pipeline
    if testing_with_inference:
        output_path, results_list = run_inference(run_args)
        
        # Convert keypoint data to coco format
        coco_data = halpe26_to_coco(results_list)
        
    else:
        coco_data = load_json(INPUT_PATH)

    # Step 2: Split into cycles
    print("Splitting the data into cycles...")
    cycle_data = split_into_cycles(coco_data, run_args, visualize=False)

    # Load the checkpoint to the model as we need it for certain data
    print("Loading Model...")
    checkpoint = torch.load(run_args.CLS_GEAR.MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]
    
    # convert to numpy arrays and stack them together
    input_data = []
    for cycle in cycle_data.values():

        # Extract joint data as (num_joints, time_steps)
        cycle_data_array = [np.array(cycle[joint], dtype=np.float32) for joint in custom_params["choosen_joints"]]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data_array)  # Shape: (num_joints, time_steps)

        input_data.append(cycle_tensor)

    # Step 3: Classify user cycles
    
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
    if run_args.CLS_GEAR.NETTYPE.lower() == "lstm":

        print("Initializing LSTM...")
        net = LSTMNet(input_channels, 
                    custom_params["hidden_size"], 
                    output_channels, 
                    custom_params["num_layers"], 
                    custom_params["dropout"])

    else:  # MLP

        print("Initializing MLP...")
        net = SimpleMLP(input_channels, 
                        custom_params["hidden1"], 
                        custom_params["hidden2"], 
                        output_channels)

    # Load weights into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    print(f"Loaded model with input={input_channels}, output={output_channels}")
    
    # reverse label dict to get predictions
    reversed_labels = {v: k for k, v in custom_params["labels"].items()}
    
    # classify each cycle
    for i, cycle_input in enumerate(input_data):
        # this is done in the dataloader 
        cycle_input = cycle_input.T # (12, 97) => (97, 12), 12 joints, 97 timesteps
        # add batch size
        cycle_input = cycle_input.unsqueeze(0)
        
        if run_args.CLS_GEAR.NETTYPE.lower() == "mlp":
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
            expert_path = "./data/expert_data/expert_cycles_gear2.json"
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
        
        dtw_comparisons, path, expert_cycle = compare_selected_cycles(expert_data, cycle, joints, run_args.VIDEO_PATH, run_args.DTW.VIS_VID_PATH, visualize=False)
        # TODO try DTW with smoothed signals?
        dtw_comparisons = compare_selected_cycles(expert_data, cycle, joints, run_args.VIDEO_PATH, run_args.DTW.VIS_VID_PATH, visualize=run_args.DTW.VIS_VIDEO)

        # Step 5: Give feedback

        direction = expert_cycle.get("Direction")
        if direction == "front":
            # Joint 1 and 2 create one line, joint 3 and 4 another line. 
            joints_lines = [("RShoulder", "LShoulder", "RHip", "LHip")]
            joint_angles = [("")]
        elif direction == "left":
            joints_lines = [("RAnkle", "RKnee", "Hip", "Neck")]
        elif direction == "right":
            joints_lines = [("LAnkle", "LKnee", "Hip", "Neck")]

        # Get the lines 
        user_lines, _ = extract_multivariate_series_for_lines(cycle, joints_lines, run_args)
        expert_lines, _ = extract_multivariate_series_for_lines(expert_cycle, joints_lines, run_args)
        
        # Match using DTW and calculate difference in angle between the lines
        diff_user_expert = calculate_differences(user_lines, expert_lines, path)
        # Flatten because it is in shape [array([value]), [array([value]), ...]
        diff_user_expert = [item[0] for item in diff_user_expert]

        #TODO set param?
        lean_threshold = 0.5
        print(np.mean(diff_user_expert))
        if np.abs(np.mean(diff_user_expert)) > lean_threshold:
            print("Not parallel shoulder and hips")


        # Plotting
        # TODO make parameter?
        if True:
            plot_lines(
                f'output/diff_shoulder_hips_{i}.png', 
                'Difference between user and expert with DTW', 
                'Time step', 
                'Angle (Degrees)', 
                diff_user_expert,  # Positional argument for *line_data
                labels=['Difference between user and expert'], 
                colors=['b'])

            plot_lines(
                f'output/user_shoulder_hips_{i}.png',
                'Plot of Array Data', 
                'Time step', 
                'Angle (Degrees)',  
                user_lines,  # Positional argument for *line_data
                expert_lines,  # Additional positional argument for *line_data
                labels=['User', 'Expert'])

        
        user_start_frame = cycle.get("Start_frame")
        # Loops through the DTW match pair and shows lines on user video
        print(path)
        for i, (frame1, frame2) in enumerate(path):
            user_frame = extract_frame(run_args.VIDEO_PATH, frame1 + user_start_frame)
            print(frame1, frame2)
            # TODO Make this a parameter?
            if True:
                expert_start_frame = expert_cycle.get("Start_frame")
                print("Experti video start frame", expert_start_frame)
                expert_video = os.path.join(run_args.DTW.VIS_VID_PATH, "DJI_00" + expert_cycle.get("Video") + ".mp4")
                expert_frame = extract_frame(expert_video, frame2 + expert_start_frame)
                print("Expert video frame:", frame2 + expert_start_frame)
                user_frame = cv2.addWeighted(user_frame, 00, expert_frame, 1, 0)
            # TODO Fix this colour conversion?
            user_frame = cv2.cvtColor(user_frame, cv2.COLOR_RGB2BGR)
            
            user_frame = draw_lines_and_text(user_frame, cycle, joints_lines, frame1, frame2, expert_cycle, 
                                      user_lines, expert_lines, diff_user_expert, i, run_args)

            cv2.imshow("User video", user_frame)
        
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()