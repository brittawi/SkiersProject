import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it
from utils.load_data import load_json
from utils.classify_angle import classify_angle

"""
This is just a test file to chedk if the classify_angle function works as expected.
ANNO_DIR: Directory where the annotations are stored, direct output from AlphaPose or annotated.
video_ids: List of tuples with video_id and the expected label (Front, Left, Right).
"""

def classify():
    ANNO_DIR = "e:/SkiProject/annotations_test_DJI_0044/After_Mixed_level_output/coco_json"
    video_ids = [(58, "Front"), (59, "Front"), (62, "Front"), (63, "Front"), (64, "Right"), (65, "Left"), (68, "Right"), (72, "Left"), (75, "Left"), (84, "Front"), (87, "Right"), (92, "Front"), (94, "Left"), (113, "Right"), (127, "Left"), (128, "Right")]
    #video_id ="68"
    #video_ids = [(58, "Front"), (59, "Front"), (62, "Front"), (63, "Front")]
    #video_ids = [(87, "Right")]
    correct = 0
    for video_id in video_ids:
        file_path = os.path.join(ANNO_DIR, f"DJI_{int(video_id[0]):04d}_coco.json")
        print(f"File path: {file_path}")
        # Load JSON data
        data = load_json(file_path)
        angle = classify_angle(data)
        if angle == video_id[1]:
            correct += 1
        print(f"Video id:{video_id[0]} | Angle: {angle} | Label: {video_id[1]}")
    acc = correct / len(video_ids)
    print(f"Accuracy: {acc}")


if __name__ == '__main__':
    classify()

