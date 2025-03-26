import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

import json
from utils.config import update_config
from alphapose.scripts.demo_inference import run_inference
from utils.annotation_format import halpe26_to_coco
from utils.feedback_utils import choose_id
import re
import cv2

def visualize_tracking(input_video_path, output_video_path, coco_annotations):
    # Define keypoint colors
    KEYPOINT_COLOR = (0, 255, 0)  # Green
    LINE_COLOR = (0, 0, 255)      # Red
    LINE_THICKNESS = 2
    KEYPOINT_RADIUS = 5
    KEYPOINT_THICKNESS = -1  # Filled circle

    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # Adjust fps as necessary

    # A function to draw keypoints and skeleton on each frame
    def draw_keypoints_and_skeleton(frame, keypoints, skeleton):
        # Iterate over the keypoints and draw them
        for i in range(0, len(keypoints), 3):  # Step of 3 (x, y, visibility)
            x, y, v = keypoints[i:i+3]
            if v > 0:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR, KEYPOINT_THICKNESS)
        
        # Draw the skeleton lines between connected keypoints
        # for pair in skeleton:
        #     start_idx, end_idx = pair
        #     print(start_idx, end_idx)
        #     print(keypoints[start_idx * 3: start_idx * 3 + 3])
        #     x1, y1, v1 = keypoints[start_idx * 3: start_idx * 3 + 3]
        #     x2, y2, v2 = keypoints[end_idx * 3: end_idx * 3 + 3]
        #     if v1 > 0 and v2 > 0:  # Only draw the line if both keypoints are visible
        #         cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), LINE_COLOR, LINE_THICKNESS)

    # Map annotations by image_id for quick lookup
    annotations_by_image_id = {}
    for annotation in coco_annotations['annotations']:
        if annotation['image_id'] not in annotations_by_image_id:
            annotations_by_image_id[annotation['image_id']] = []
        annotations_by_image_id[annotation['image_id']].append(annotation)

    # Iterate through all frames (image_ids) in the video
    frame_id = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Check if there are any annotations for this frame (image_id)
        if frame_id + 1 in annotations_by_image_id:  # Image IDs are 1-based
            annotations = annotations_by_image_id[frame_id + 1]
            
            for annotation in annotations:
                keypoints = annotation['keypoints']
                skeleton = coco_annotations['categories'][annotation['category_id'] - 1]['skeleton']
                
                # Draw the keypoints and skeleton on the current frame
                draw_keypoints_and_skeleton(frame, keypoints, skeleton)

        # Write the frame (with or without annotations) to the output video
        out.write(frame)

        # Increment the frame ID
        frame_id += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video processing complete. The output video has been saved.")


def main():
    # Load config
    run_args = update_config("./feedback_system/pipe_test.yaml") # TODO Testing set up fix for full pipeline
    #folder_path = os.path.dirname(run_args.VIDEO_PATH)
    folder_path = "C:\\awilde\\britta\\LTU\\SkiingProject\\SkiersProject\\Data\\NewData\\Film2025-02-22\\"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp4") or file_name.endswith(".MP4"):
            full_path = os.path.join(folder_path, file_name)
            run_args.VIDEO_PATH = full_path
            _, results_list = run_inference(run_args)
    
            # file_path = "C:\\Users\\britt\\AlphaPose\\examples\\res\\alphapose-results.json"
            # with open(file_path, "r") as f:
            #     results_list = json.load(f)
            
            # choose the ID that we want to track
            #tracker_id = choose_id(results_list, full_path)
                    
            output_folder = "./data/alphapose_output/coco_json"
            # filter out id of video
            file_id = re.search(r'\d+', os.path.basename(full_path)).group() 
            # Strip leading zeros, convert to an integer and back to a string
            file_id = str(int(file_id))
            
            #file_name_json = os.path.basename(full_path).split(".")[0] + "_coco.json"
            file_name_json = file_id + ".json"
            output_path = os.path.join(output_folder, file_name_json)
            # choosing ID with highest occurences
            coco_data, tracked_id, save_verification = halpe26_to_coco(results_list)
            print(f"ID {tracked_id} was tracked")
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=4)
                
            # for verification
            if save_verification:
                output_path_verification = "./data/alphapose_output/" + file_id + "verif.mp4"
                visualize_tracking(full_path, output_path_verification, coco_data)


if __name__ == '__main__':
    main()