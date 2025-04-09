import numpy as np
import cv2
import matplotlib.pyplot as plt
from dtaidistance import dtw, dtw_visualisation
import os
from utils.preprocess_signals import smooth_signal, normalize_signal
from utils.frame_extraction import extract_frame

def compute_angle(p1, p2, p3):
    """Computes the angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(dot_product / norm_product) * (180.0 / np.pi)

# TODO do not need anymore I think?! For single signal
def extract_angle_series(cycle_data, joint1, joint2, joint3, compute_angle_func=None):
    """Extracts time-series data for an angle between three joints from a cycle."""
    if compute_angle_func == None:
        compute_angle_func = compute_angle
    angles = []
    frames = []
    for i in range(len(cycle_data[joint1 + "_x"])):
        p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
        p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
        p3 = (cycle_data[joint3 + "_x"][i], cycle_data[joint3 + "_y"][i])
        angles.append(compute_angle_func(p1, p2, p3))
        frames.append(i)
    return np.array(angles), frames

# function to extract several angles at once
def extract_multivariate_series(cycle_data, joint_triplets, run_args):
    """Extracts multivariate time-series data for multiple angles from a cycle."""
    all_angles = []
    frames = []

    joints_list = [joint for tuple_joints in joint_triplets for joint in tuple_joints]
    if run_args.DTW.GAUS_FILTER:
        cycle_data = smooth_cycle(cycle_data, joints_list, sigma=run_args.DTW.SIGMA_VALUE)

    for i in range(len(cycle_data[joint_triplets[0][0] + "_x"])):
        angles = []
        for joint1, joint2, joint3 in joint_triplets:
            p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
            p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
            p3 = (cycle_data[joint3 + "_x"][i], cycle_data[joint3 + "_y"][i])
            angles.append(compute_angle(p1, p2, p3))
        all_angles.append(angles)
        frames.append(i)
    return np.array(all_angles), frames

def smooth_cycle(cycle_data, joints, sigma=2, norm=False):
    smoothed_cycle_data = {}
    # apply smoothing first
    if sigma > 0:
        for joint, series in cycle_data.items():
            # Check if the joint is in the joints list if we should smooth
            joint_base = joint.replace("_x", "").replace("_y", "")
            joint_ref = joint.replace("_x_ref", "").replace("_y_ref", "") # Need to include reference
            if joint_base in joints or joint_ref in joints:
                if norm:
                    series = normalize_signal(series)
                smoothed_series = smooth_signal(series, sigma)
                smoothed_cycle_data[joint] = smoothed_series
    return smoothed_cycle_data

def extract_keypoint_series(cycle_data, joints, sigma=2):
    smoothed_cycle_data = smooth_cycle(cycle_data, joints, sigma)
    all_keypoints = []
    frames = []
    for i in range(len(smoothed_cycle_data[joints[0] + "_x"])):
        keypoints = []
        for joint in joints:
            keypoints.append(smoothed_cycle_data[joint + "_x"][i])
            keypoints.append(smoothed_cycle_data[joint + "_y"][i])
        all_keypoints.append(keypoints)
        frames.append(i)
    return np.array(all_keypoints), frames
        
# function to overlay 2 frames
# TODO are we actually using this??!
def overlay_frames_loop(user_video, expert_video, path, cycle1_start_frame, cycle2_start_frame, series1, series2):
    """Overlays frames one by one with OpenCV and displays frame numbers."""
    if not path:
        print("No frames to overlay.")
        return
    
    for frame1, frame2 in path:

        f1 = extract_frame(user_video, frame1 + cycle1_start_frame)
        f2 = extract_frame(expert_video, frame2 + cycle2_start_frame)
        if f1 is not None and f2 is not None:
            overlay = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            cv2.putText(overlay, f"Frame 1: {frame1}, keypoints: {series1[frame1]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, f"Frame 2: {frame2}, keypoints: {series2[frame2]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Overlayed Frames", overlay)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

# function to compare cycles based on DTW
def compare_selected_cycles(expert_data, cycle, joint_triplets, user_video, video_path, use_keypoints = True, visualize=False):
    """Compares two selected movement cycles using DTW and optionally visualizes the alignment."""
    
    if use_keypoints:
        extract_signals = extract_keypoint_series
        if any(isinstance(item, tuple) for item in joint_triplets):
            raise ValueError("Tuples of joints were given, but if keypoints should be used for DTW only joints should be given")
    else:
        extract_signals = extract_multivariate_series
        if not all(isinstance(item, tuple) for item in joint_triplets):
            raise ValueError("Only joints were given, but for angles tuple of joints need to be given")
    
    dtw_results = {}
    best_dist = float("inf")
    closest_cycle = {}
    expert_video = ""
    
    series_user, frames_user = extract_signals(cycle, joint_triplets)
    
    for cycle_key, expert_cycle in expert_data.items():
        
        series_expert, frames_expert = extract_signals(expert_cycle, joint_triplets)
        
        dist = dtw.distance(series_user, series_expert, use_ndim=True)
        if dist < best_dist:
            best_dist = dist
            closest_cycle = expert_cycle
            #dtw_results = {f"{cycle1_key} vs {cycle2_key} (Multivariate DTW)": dist}
    
    # # TODO direction is only there for test purposes
    # print(f"Choosen expert cyle direction: {closest_cycle['Direction']}, gear: {closest_cycle['Label']}")
    # print(f"User cycle label: {cycle['Label']}")
    
    # compute dtw for closest cycle
    series_expert, frames_expert = extract_signals(closest_cycle, joint_triplets)
    path = dtw.warping_path(series_user, series_expert, use_ndim=True)
    
    # get start frames
    user_start_frame = cycle.get("Start_frame")
    expert_start_frame = closest_cycle.get("Start_frame")
    
    # get video path
    expert_video = os.path.join(video_path, "DJI_00" + closest_cycle.get("Video") + ".mp4")

    
    if visualize:
        overlay_frames_loop(user_video, expert_video, path, user_start_frame, expert_start_frame, series_user, series_expert)

    return dtw_results, path, closest_cycle