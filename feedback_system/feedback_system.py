import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles


def main():
    # TODO put in config file?! 
    
    # Step 1: Get Keypoints from AlphaPose
    
    # Step 2: Split into cycles
    
    # Step 3: Classify user cycles
    # Load Model
    

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
    
    dtw_comparisons = compare_selected_cycles(expert_data, user_data, joints, user_video, video_path, visualize=True)
    

if __name__ == '__main__':
    main()