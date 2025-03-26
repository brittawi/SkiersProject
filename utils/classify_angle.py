import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it
from utils.dtw import extract_angle_series
from utils.split_cycles import process_data, convert_keypoint_format
import numpy as np

def compute_angle_360(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_norm, v2_norm)
    cross_product = np.cross(v1_norm, v2_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle precision errors
    angle_deg = np.degrees(angle_rad)
    if cross_product < 0:
        angle_deg = 360 - angle_deg  # Convert to reflex angle (clockwise)

    return angle_deg

def classify_angle(data):
    """
    Classify the video as Front, Left or Right based on the knee position and the angle of the left leg.
    """
    CHOOSEN_REF = "Hip"
    keypoint_labels = data["categories"][0]["keypoints"]
    keypoint_dict = {}
    for i, label in enumerate(keypoint_labels):
        keypoint_dict[label] = {"x": [], "y": [], "index" : i}

    # Process dataset, get keypoints and frames
    _, keypoints = process_data(data, keypoint_dict, CHOOSEN_REF)

    keypoints = convert_keypoint_format(keypoints, CHOOSEN_REF)
    l_knee_count = 0
    total_timesteps = len(keypoints["LKnee_x"])


    for i, (rx, lx) in enumerate(zip(keypoints["RKnee_x"], keypoints["LKnee_x"])):
        #print(f"frame {i} | lx: {lx} | rx: {rx} | diff: {lx - rx} | {lx > rx}")
        if lx > rx:
            l_knee_count += 1
    
    l_knee_ratio = l_knee_count / total_timesteps

    left_leg_angle = extract_angle_series(keypoints, "LHip", "LKnee", "LAnkle", compute_angle_360)

    if l_knee_ratio > 0.9:
        return "Front"
    elif np.mean(left_leg_angle[0]) > 180:
        return "Left"
    elif np.mean(left_leg_angle[0]) < 180:
        return "Right"