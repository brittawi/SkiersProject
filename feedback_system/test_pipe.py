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


def main():
    # Load config
    run_args = update_config("./feedback_system/pipe_test.yaml") # TODO Testing set up fix for full pipeline
    #folder_path = os.path.dirname(run_args.VIDEO_PATH)
    folder_path = "E:\SkiProject\Cut_videos\Edited"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp4"):
            full_path = os.path.join(folder_path, file_name)
            run_args.VIDEO_PATH = full_path
            _, results_list = run_inference(run_args)
            output_folder = "./data/alphapose_output/coco_json"
            file_name_json = os.path.basename(full_path).split(".")[0] + "_coco.json"
            output_path = os.path.join(output_folder, file_name_json)
            coco_data = halpe26_to_coco(results_list)
            print(file_name_json)
            print(output_path)
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=4)


    
    # _, results_list = run_inference(run_args)
    # output_path = "./data/alphapose_output/coco_json"
    # print(len(results_list))
    # coco_data = halpe26_to_coco(results_list)
    # with open(output_path, 'w') as f:
    #     json.dump(coco_data, f, indent=4)







if __name__ == '__main__':
    main()