from utils.load_json import load_json
from utils.dtw import compare_selected_cycles

def main():
    # TODO put in config file?! 
    
    # Step 1: Get Keypoints from AlphaPose
    
    # Step 2: Split into cycles
    
    # Step 3: Classify user cycles
    # Load Model

    # Step 4: Based on classification use DTW
    
    # Load cycle data
    file_path = "../classification/cycle_splits/labeled_data/labeled_cycles_17_cut.json" # r->l gear 3, cycle 5
    #file_path = "../classification/cycle_splits/labeled_data/labeled_cycles_18_cut.json" # l->r gear 3, cycle 6
    #file_path = "../classification/cycle_splits/labeled_data/labeled_cycles_38.json" # front gear 3, cycle 5
    cycle_to_compare = "Cycle 5"
    video_path = r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_0018_cut.mp4"  # Path to the corresponding video file
    data = load_json(file_path)
    user_data = data.get(cycle_to_compare)
    
    # Load expert data
    expert_path = "./expert_cycles_gear3.json"
    expert_video_paths = [r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_0017_cut.mp4",
                   r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_0009.mp4",
                   r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\selectedData\DJI_0018_cut.mp4"]
    expert_data = load_json(expert_path)

    # Define joint triplets for angle comparisons
    joint_triplets = [("RHip", "RKnee", "RAnkle"), ("LHip", "LKnee", "LAnkle"), ("RShoulder", "RElbow", "RWrist"), ("LShoulder", "LElbow", "LWrist")]
    
    dtw_comparisons = compare_selected_cycles(expert_data, user_data, joint_triplets, expert_video_paths, video_path, visualize=True)
    

if __name__ == '__main__':
    main()