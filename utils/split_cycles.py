from utils.load_data import load_json
from utils.preprocess_signals import smooth_signal, normalize_signal, compute_relative_keypoints

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema  # Local minima detection

def convert_keypoint_format(keypoints, chosen_ref):
    keypoint_values = {}
    for joint, values in keypoints.items(): 
        if joint == chosen_ref:
            key_x, key_y = joint + "_x_ref", joint + "_y_ref"
        else:
            key_x, key_y = joint + "_x", joint + "_y"
        keypoint_values[key_x], keypoint_values[key_y] = values["x"], values["y"]
    return keypoint_values
    


def split_into_cycles(data, run_args, video_angle, visualize=False):

    if video_angle == "Front":
        chosen_joint = run_args.DTW.SPLIT.FRONT.CHOOSEN_JOINT
        extrema = run_args.DTW.SPLIT.FRONT.CHOOSEN_EXTREMA
    elif video_angle == "Left":
        chosen_joint = run_args.DTW.SPLIT.LEFT.CHOOSEN_JOINT
        extrema = run_args.DTW.SPLIT.LEFT.CHOOSEN_EXTREMA
    elif video_angle == "Right":
        chosen_joint = run_args.DTW.SPLIT.RIGHT.CHOOSEN_JOINT
        extrema = run_args.DTW.SPLIT.RIGHT.CHOOSEN_EXTREMA
    else:
        raise ValueError(f"Invalid video angle: {video_angle}")

    if extrema == "min":
        extrema_func = np.less
    elif extrema == "max":
        extrema_func = np.greater
    else:
        raise ValueError(f"Invalid extrema value: {extrema}")
    
    # Extract keypoint labels (assumes same structure across files)
    keypoint_labels = data["categories"][0]["keypoints"]
    
    
    # create a dict where we save the keypoint values per joint and indices for each joint
    keypoint_dict = {}
    for i, label in enumerate(keypoint_labels):
        keypoint_dict[label] = {"x": [], "y": [], "index" : i}

    # get keypoints and frames
    frames, keypoints = process_data(data, keypoint_dict, run_args.DTW.CHOOSEN_REF)
    
    # Normalize and smooth the choosen signal
    smoothed_normalized_keypoints = {"x": [], "y": []}
    smoothed_normalized_keypoints["x"] = normalize_signal(keypoints[chosen_joint]["x"])
    smoothed_normalized_keypoints["y"] = normalize_signal(keypoints[chosen_joint]["y"])

    smoothed_normalized_keypoints["x"] = smooth_signal(smoothed_normalized_keypoints["x"], sigma=run_args.DTW.SIGMA_VALUE)
    smoothed_normalized_keypoints["y"] = smooth_signal(smoothed_normalized_keypoints["y"], sigma=run_args.DTW.SIGMA_VALUE)
    
    # Detect local minima for X movement
    x_values = np.array(smoothed_normalized_keypoints["x"])
    x_extrema_indices = argrelextrema(x_values, extrema_func, order=run_args.DTW.ORDER)[0]  
    
    # Detect local minima for Y movement
    y_values = np.array(smoothed_normalized_keypoints["y"])
    y_extrema_indices = argrelextrema(y_values, extrema_func, order=run_args.DTW.ORDER)[0]
    
    # Plot if needed
    if visualize:
        # Plot Joint movement (X and Y in separate plots)
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)  

        # Plot choosen joint X movement
        axs[0].plot(frames, x_values, label=f"{chosen_joint} X ({file_path[-11:-5]})", color="red")
        axs[0].scatter(np.array(frames)[x_extrema_indices], x_values[x_extrema_indices], color="black", marker="o", label="Local Minima")
        axs[0].set_ylabel("Normalized X Position")
        axs[0].legend()

        # Plot choosen joint Y movement
        axs[1].plot(frames, y_values, label=f"{chosen_joint} Y ({file_path[-11:-5]})", color="blue")
        axs[1].scatter(np.array(frames)[y_extrema_indices], y_values[y_extrema_indices], color="black", marker="o", label="Local Minima")
        axs[1].set_ylabel("Normalized Y Position")
        axs[1].set_xlabel("Frame")
        axs[1].legend()

        plt.suptitle(f"Smoothed {chosen_joint} Movement with Local Minima (Gaussian σ={run_args.DTW.SIGMA_VALUE})")
        plt.tight_layout()
        plt.show()
    
    # creating structure for returning keypoints
    #keypoints = {"LAnkle" : {"x": [], "y" : []}}
    keypoint_values = {}
    for joint, values in keypoints.items(): 
        if joint == run_args.DTW.CHOOSEN_REF:
            key_x, key_y = joint + "_x_ref", joint + "_y_ref"
        else:
            key_x, key_y = joint + "_x", joint + "_y"
        keypoint_values[key_x], keypoint_values[key_y] = values["x"], values["y"]
    
    # keypoint_values = {"LAnkle_x": [], "LAnkle_y": [], ...}  
      
    current_min_index = 0  # Track local minima
    
    # choose dim 
    if run_args.DTW.CHOOSEN_DIM == "x":
        extrema_indices = x_extrema_indices
    else:
        extrema_indices = y_extrema_indices
    
    # split cycles and savee
    data_split_by_cycles = {} 
    cycle = 1   
    while current_min_index < len(extrema_indices) - 1:  # Ensure at least two minima
        start_frame = frames[extrema_indices[current_min_index]]
        cycle_values_for_joints = {}
        for joint, values in keypoint_values.items():
            cycle_vals = values[extrema_indices[current_min_index] : extrema_indices[current_min_index + 1] + 1]
            cycle_values_for_joints[joint] = cycle_vals
        
        # save Image ids for each value for mapping later
        # this is needed so that we can still find the correct image id in case we have frames missing in the middle
        cycle_values_for_joints["Image_ids"] = frames[extrema_indices[current_min_index] : extrema_indices[current_min_index + 1] + 1]
        cycle_values_for_joints["Start_frame"] = start_frame    
        data_split_by_cycles[f"Cycle {cycle}"] = cycle_values_for_joints
        cycle += 1
        current_min_index += 1
            
    return data_split_by_cycles
        
        
# Function to process data from a dataset
def process_data(data, keypoint_dict, chosen_ref):
    frames = sorted(set(anno["image_id"] for anno in data.get("annotations", [])))
    
    keypoint_movements = {}
    for joint, keypoints_per_joint in keypoint_dict.items():
        # keypoints_per_joint = {"x": [], "y": []}
        index = keypoints_per_joint["index"]
        
        for annotation in data.get("annotations", []):
            keypoints = annotation["keypoints"]
            if len(keypoints)<= 0:
                print(keypoints)
                print(annotation["image_id"])
            
            # TODO
            # for the choosen reference joint we save the absolute values, but only if it is in choosen joints
            if joint == chosen_ref:
                absolute_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    if v == 0:
                        x, y = None, None
                    absolute_keypoints.append((x, y))
                joint_x, joint_y = absolute_keypoints[index]
            # for other joints we save keypoints relative to our reference joints
            else:
                relative_keypoints = compute_relative_keypoints(keypoints, keypoint_dict[chosen_ref]["index"])
                joint_x, joint_y = relative_keypoints[index]

            keypoints_per_joint["x"].append(joint_x)
            keypoints_per_joint["y"].append(joint_y)
            
        keypoint_movements[joint] = keypoints_per_joint

    return frames, keypoint_movements  