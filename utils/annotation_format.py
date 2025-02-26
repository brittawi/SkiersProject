from datetime import datetime
from collections import defaultdict
import json

# ChatGPT generated!
def halpe26_to_coco(halpe26_data):
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    coco_data = {
        "info": {
            "description": "Crosscountry skiing",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "LTU",
            "date_created": current_date
        },
        "licenses": [
            {
                "id": 0,
                "name": "",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "Nose", "LEye", "REye", "LEar", "REar", 
                    "LShoulder", "RShoulder", "LElbow", "RElbow", 
                    "LWrist", "RWrist", "LHip", "RHip", 
                    "LKnee", "RKnee", "LAnkle", "RAnkle", 
                    "Head", "Neck", "Hip", 
                    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", 
                    "LHeel", "RHeel"
                ],
                "skeleton": [
                    [25,23],[26,24],[3,1],[14,16],[19,6],[9,11],[15,17],[6,8],[20,13],[18,19],[16,25],[20,19],[2,4],[1,2],[17,26],
                    [25,21],[7,9],[26,22],[20,12],[12,14],[3,5],[19,7],[8,10],[13,15]
                    ]
            }
        ]
    }

    # Map to track filenames and their assigned image IDs
    filename_to_id = {}
    annotations_grouped_by_image = defaultdict(list)
    annotation_id = 1

    for annotation in halpe26_data:
        file_name = annotation["image_id"]

        # Assign an image_id to each unique file_name
        if file_name not in filename_to_id:
            new_image_id = len(filename_to_id) + 1
            filename_to_id[file_name] = new_image_id
            coco_data["images"].append({
                "id": new_image_id,
                "file_name": file_name,
                "width": 1920,  # Replace with actual width if available
                "height": 1080,  # Replace with actual height if available
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })

        # Use the assigned image_id for this file
        image_id = filename_to_id[file_name]
        
        # Extract score for filtering
        score = annotation["score"]
        # Group annotations by image_id
        annotations_grouped_by_image[image_id].append((annotation, score))
        
    # Filter annotations to keep only the one with the highest score per image_id
    for image_id, annotations in annotations_grouped_by_image.items():

        # Find the annotation with the highest score
        best_annotation, _ = max(annotations, key=lambda x: x[1])
        
        # Process keypoints
        keypoints = best_annotation.get("keypoints", [])
        processed_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, score = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if x is None or y is None:  # Check for None values
                x, y = 0, 0  # Default to (0, 0) if missing
            z = 1  # Default z-coordinate
            processed_keypoints.extend([x, y, z])

        # Validate bbox
        bbox = best_annotation.get("box", [0, 0, 0, 0])
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]  # Default bbox if invalid

        # Add annotation
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": processed_keypoints,
            "num_keypoints": 26,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "segmentation": []
        })
        annotation_id += 1

    return coco_data
    # # Save the updated COCO JSON file
    # with open(output_file, 'w') as f:
    #     json.dump(coco_data, f, indent=4)
