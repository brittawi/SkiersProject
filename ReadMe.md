# Project Title

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)

## Project Description (Britta)
TODO short description of project

TODO put Halpe dataset here as well?!

## Installation (Emil)
TODO how to install AlphaPose and how to use it

## Usage
TODO folder structure, where to find what and how to use it

### Classification
    .
    ├── cycle_splits                # Files to split video into cycles and plot these
    │   ├── labeled data            # Cycles labeled with corresponding gear
    │   ├── lable_cycles.ipynb      # Script to split video into cycles and label these
    ├── training                    # Scripts for Training a classification model (MLP, LSTM)

This folder contains files that can be used to split a video of a crosscountry skier into cycles. 

#### Label Cycles (Britta)

With the file `lable_cycles.ipynb` a video of a crosscountry skier can be split into cycles and the cycles can be labeled with the corresponding gear. To do so the video and the keypoint annotations are needed. The keypoint annotations need to follow the COCO format and therefore contain the following section:

```
{
    "info": {...},
    "licenses": [...],
    "images": [...],
    "annotations": [...],
    "categories": [...]
}
```
We are using the following keypoints: `["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle","Head","Neck","Hip","LBigToe","RBigToe","LSmallToe","RSmallToe","LHeel","RHeel"]`

Under Config you can modify the following parameters:
|Param  | Description   |
|------ |---------------|
|CHOOSEN_JOINT | |
|CHOOSEN_REF| |
|CHOOSEN_DIM| |
|sigma_value| |
|order| |
|VIDEO_DIR| |
|ANNO_DIR| |
|video_id| |

#### Training (Emil)

#### Create Annotations (Emil)

#### Other models (Britta)
(write about Installation of MMPose?)



## Contribution
TODO not sure if we even need this  

## TODO
- [ ] Feedback system
- [ ] Crossvalidation

