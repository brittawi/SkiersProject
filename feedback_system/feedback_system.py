import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles, extract_multivariate_series, extract_angle_series, extract_multivariate_series_for_lines, calculate_differences, extract_frame
from utils.nets import LSTMNet, SimpleMLP
from utils.config import update_config
from utils.split_cycles import split_into_cycles
from utils.preprocess_signals import *
from alphapose.scripts.demo_inference import run_inference

import torch
import numpy as np
import cv2

# TODO put in different config??
# Type of network that we want to use for the classification
NETWORK_TYPE = "MLP"
# Model path where we want to load the model from
MODEL_PATH = "./pretrained_models/best_model_2025_02_25_15_55_lr0.0001_seed42.pth"
# TODO this is just for test purposes. It is not needed anymore once we get AlphaPose to work, as we do not need to read in the annotated data then
ID = "38"
# INPUT_PATH = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\Annotations\\" + ID + ".json"
# INPUT_VIDEO = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_00" + ID + ".mp4"
INPUT_PATH = os.path.join("E:\SkiProject\AnnotationsByUs", ID[:2] + ".json")
INPUT_VIDEO = r"E:\SkiProject\Cut_videos\DJI_00" + ID + ".mp4"
# path to where all videos are stored
# video_path = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData"
video_path = r"E:\SkiProject\Cut_videos"

# TODO Move to utils/dtw
def get_line_points(user_cycle, joints_list, frame, expert_cycle = None):
    points = []
    for joints in joints_list:
        #TODO Get ref from cfg/other way?
        for joint in joints:
            if expert_cycle == None:
                p_x = int(user_cycle.get(joint + "_x")[frame] + user_cycle.get("Hip_x_ref")[frame])
                p_y = int(user_cycle.get(joint + "_y")[frame] + user_cycle.get("Hip_y_ref")[frame])
            else:
                p_x = int(expert_cycle.get(joint + "_x")[frame] + user_cycle.get("Hip_x_ref")[frame])
                p_y = int(expert_cycle.get(joint + "_y")[frame] + user_cycle.get("Hip_y_ref")[frame])
            points.append((p_x,p_y))
    return points

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
        
        dtw_comparisons, path, expert_cycle = compare_selected_cycles(expert_data, cycle, joints, INPUT_VIDEO, video_path, visualize=False)

        # Step 5: Give feedback
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

        Start with parallel shoulder and hip lines
        """

        # Feedback angle

        # sample_cycle_series, frames_user = extract_angle_series(cycle, joint_triplets)

        # # TODO JUST FOR PLOTTING REMOVE LATER
        import matplotlib.pyplot as plt
        # # Extract the array and the list from the tuple
        # data_array, x_values = sample_cycle_series, frames_user

        # # Create the plot
        # plt.figure(figsize=(10, 6))
        # for i in range(data_array.shape[1]):
        #     plt.plot(x_values, data_array[:, i], label=f'Line {i+1}')

        # plt.xlabel('X values')
        # plt.ylabel('Y values')
        # plt.title('Plot of Array Values vs X List')
        # plt.legend()
        # # Save the plot to a file
        # plt.savefig('output/array_vs_list_plot.png')


        # #print(path)
        # print(len(path))

        # Joint 1 and 2 create one line, joint 3 and 4 another line. 
        shoulder_hip_joints = [("RShoulder", "LShoulder", "RHip", "LHip")]

        # Get the lines 
        user_lines, _ = extract_multivariate_series_for_lines(cycle, shoulder_hip_joints)
        expert_lines, _ = extract_multivariate_series_for_lines(expert_cycle, shoulder_hip_joints)
        
        # Match using DTW and calculate difference in angle between the lines
        diff_user_expert = calculate_differences(user_lines, expert_lines, path)

        # Flatten because it is in shape [array([value]), [array([value]), ...]
        diff_user_expert = [item[0] for item in diff_user_expert]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(diff_user_expert, linestyle='-', color='b', label='Values')
        plt.title('Difference between user and expert with DTW')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('output/diff_shoulder_hips.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(user_lines, linestyle='-', label='User')
        plt.plot(expert_lines, linestyle='-', label='Expert')

        plt.title('Plot of Array Data')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('output/user_shoulder_hips.png')

        user_start_frame = cycle.get("Start_frame")
        expert_start_frame = expert_cycle.get("Start_frame")
        print("User start frame: ", user_start_frame, " Expert start frame:", expert_start_frame)

        # TODO Can get from compare_selected_cycles()
        # TODO Do not need?
        expert_video = os.path.join(video_path, "DJI_00" + expert_cycle.get("Video") + ".mp4")
        """
        Get user video frame
        Plot user line of shoulder + hips
        Overlay and also plot line of expert shoulder + hips
        """   
        for i, (frame1, frame2) in enumerate(path):
            user_frame = extract_frame(INPUT_VIDEO, frame1 + user_start_frame)
            # TODO Fix this colour conversion
            user_frame = cv2.cvtColor(user_frame, cv2.COLOR_RGB2BGR)
            
            # Draw the lines on the skier
            user_points = get_line_points(cycle, shoulder_hip_joints, frame1)
            cv2.line(user_frame, user_points[0], user_points[1], color=(255, 0, 0), thickness=2) # Line first pair
            cv2.line(user_frame, user_points[2], user_points[3], color=(255, 0, 0), thickness=2) # Line second pair
            expert_points = get_line_points(cycle, shoulder_hip_joints, frame2, expert_cycle)
            cv2.line(user_frame, expert_points[0], expert_points[1], color=(255, 255, 0), thickness=2) # Line first pair
            cv2.line(user_frame, expert_points[2], expert_points[3], color=(255, 255, 0), thickness=2) # Line second pair

            # Write degree of parallel lines
            text_origin = (int(cycle.get("Hip_x_ref")[frame1]), int(cycle.get("Hip_y_ref")[frame1]))
            # Text offset from origin point
            x_offset = 30
            y_offset = 20
            # Get the sizes of the text boxes
            (text_width_user, text_height_user), _ = cv2.getTextSize("User angle between shoulder/hip:" + str(user_lines[frame1][0]), cv2.FONT_HERSHEY_PLAIN, 1, 1)
            (text_width_expert, text_height_expert), _ = cv2.getTextSize("Expert angle between shoulder/hip:" +str(expert_lines[frame2][0]), cv2.FONT_HERSHEY_PLAIN, 1, 1)
            (text_width_diff, text_height_diff), _ = cv2.getTextSize("Difference:" +str(diff_user_expert[i]), cv2.FONT_HERSHEY_PLAIN, 1, 1)

            # Draw background rectangles for the text (adjust size for better fit)
            cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] - text_height_user - 5), 
                        (text_origin[0] + x_offset + text_width_user + 5, text_origin[1] + 5), (0, 0, 0), -1)  # Black box for user text
            cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] + y_offset - text_height_expert - 5), 
                        (text_origin[0] + x_offset + text_width_expert + 5, text_origin[1] + y_offset + 5), (0, 0, 0), -1)  # Black box for expert text
            cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] - y_offset - text_height_diff - 5), 
                        (text_origin[0] + x_offset + text_width_diff + 5, text_origin[1] - y_offset + 5), (0, 0, 0), -1)  # Black box for difference text
            cv2.putText(user_frame, "User angle between shoulder/hip: " + str(user_lines[frame1][0]), (text_origin[0] + x_offset, text_origin[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            cv2.putText(user_frame, "Expert angle between shoulder/hip: " + str(expert_lines[frame2][0]), (text_origin[0] + x_offset, text_origin[1] + y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
            cv2.putText(user_frame, "Difference: " + str(diff_user_expert[i]), (text_origin[0] + x_offset, text_origin[1] - y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))


            cv2.imshow("User video", user_frame)
            
             

        
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()






 
    
    

if __name__ == '__main__':
    main()