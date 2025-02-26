from utils.load_data import load_json
from utils.preprocess_signals import smooth_signal, normalize_signal, compute_relative_keypoints

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema  # Local minima detection


def split_into_cycles(data, run_args, visualize=False):
    
    # Extract keypoint labels (assumes same structure across files)
    keypoint_labels = data["categories"][0]["keypoints"]
    
    # keypoint_indices = {label: index for index, label in enumerate(keypoint_labels)}
    
    # create a dict where we save the keypoint values per joint and indices for each joint
    keypoint_dict = {}
    for i, label in enumerate(keypoint_labels):
        keypoint_dict[label] = {"x": [], "y": [], "index" : i}

    # get keypoints and frames
    frames, keypoints = process_data(data, keypoint_dict, run_args)
    
    # Normalize and smooth the choosen signal
    smoothed_normalized_keypoints = {"x": [], "y": []}
    smoothed_normalized_keypoints["x"] = normalize_signal(keypoints[run_args.DTW.CHOOSEN_JOINT]["x"])
    smoothed_normalized_keypoints["y"] = normalize_signal(keypoints[run_args.DTW.CHOOSEN_JOINT]["y"])

    smoothed_normalized_keypoints["x"] = smooth_signal(smoothed_normalized_keypoints["x"], sigma=run_args.DTW.SIGMA_VALUE)
    smoothed_normalized_keypoints["y"] = smooth_signal(smoothed_normalized_keypoints["y"], sigma=run_args.DTW.SIGMA_VALUE)
    
    # Detect local minima for X movement
    x_values = np.array(smoothed_normalized_keypoints["x"])
    x_min_indices = argrelextrema(x_values, np.less, order=run_args.DTW.ORDER)[0]  
    
    # Detect local minima for Y movement
    y_values = np.array(smoothed_normalized_keypoints["y"])
    y_min_indices = argrelextrema(y_values, np.less, order=run_args.DTW.ORDER)[0]
    
    # Plot if needed
    if visualize:
        # Plot Joint movement (X and Y in separate plots)
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)  

        # Plot choosen joint X movement
        axs[0].plot(frames, x_values, label=f"{run_args.DTW.CHOOSEN_JOINT} X ({file_path[-11:-5]})", color="red")
        axs[0].scatter(np.array(frames)[x_min_indices], x_values[x_min_indices], color="black", marker="o", label="Local Minima")
        axs[0].set_ylabel("Normalized X Position")
        axs[0].legend()

        # Plot choosen joint Y movement
        axs[1].plot(frames, y_values, label=f"{run_args.DTW.CHOOSEN_JOINT} Y ({file_path[-11:-5]})", color="blue")
        axs[1].scatter(np.array(frames)[y_min_indices], y_values[y_min_indices], color="black", marker="o", label="Local Minima")
        axs[1].set_ylabel("Normalized Y Position")
        axs[1].set_xlabel("Frame")
        axs[1].legend()

        plt.suptitle(f"Smoothed {run_args.DTW.CHOOSEN_JOINT} Movement with Local Minima (Gaussian σ={run_args.DTW.SIGMA_VALUE})")
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
        min_indices = x_min_indices
    else:
        min_indices = y_min_indices
    
    # split cycles and savee
    data_split_by_cycles = {} 
    cycle = 1   
    while current_min_index < len(min_indices) - 1:  # Ensure at least two minima
        start_frame = frames[min_indices[current_min_index]]
        cycle_values_for_joints = {}
        for joint, values in keypoint_values.items():
            cycle_vals = values[min_indices[current_min_index] : min_indices[current_min_index + 1] + 1]
            cycle_values_for_joints[joint] = cycle_vals
            
        cycle_values_for_joints["Start_frame"] = start_frame    
        data_split_by_cycles[f"Cycle {cycle}"] = cycle_values_for_joints
        cycle += 1
        current_min_index += 1
            
    return data_split_by_cycles
        
        
# Function to process data from a dataset
def process_data(data, keypoint_dict, run_args):
    frames = sorted(set(anno["image_id"] for anno in data.get("annotations", [])))
    
    keypoint_movements = {}
    for joint, keypoints_per_joint in keypoint_dict.items():
        # keypoints_per_joint = {"x": [], "y": []}
        index = keypoints_per_joint["index"]
        
        for annotation in data.get("annotations", []):
            keypoints = annotation["keypoints"]
            
            # TODO
            # for the choosen reference joint we save the absolute values, but only if it is in choosen joints
            if joint == run_args.DTW.CHOOSEN_REF:
                absolute_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    if v == 0:
                        x, y = None, None
                    absolute_keypoints.append((x, y))
                joint_x, joint_y = absolute_keypoints[index]
            # for other joints we save keypoints relative to our reference joints
            else:
                relative_keypoints = compute_relative_keypoints(keypoints, keypoint_dict[run_args.DTW.CHOOSEN_REF]["index"])
                joint_x, joint_y = relative_keypoints[index]

            keypoints_per_joint["x"].append(joint_x)
            keypoints_per_joint["y"].append(joint_y)
            
        keypoint_movements[joint] = keypoints_per_joint

    return frames, keypoint_movements  