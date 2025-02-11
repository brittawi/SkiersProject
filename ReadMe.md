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
### AlphaPose Installation
Follow installation from:
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md

Start with conda commands but switch to pip.

1. Create and activate conda enviroment
<br>```conda create -n alphapose python=3.7 -y```
<br>```conda activate alphapose```
2. Install pytroch version 11.3
<br> ```pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113```
3. Clone AlphaPose github and change to current working directory
<br>```git clone https://github.com/MVIG-SJTU/AlphaPose.git```
<br>```cd AlphaPose```
4. Install cython
<br>```pip install cython```
5. Run the setup python file
<br>```python setup.py build develop --user ```

To use pretrained Halpe26 model download from the [Model Zoo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md) and put the halpe26_fat_res50_256x192.pth file into into the pretrained_models folder:

    .
    ├── ...
    ├── pretrained_models
    │   └── halpe26_fat_res50_256x192.pth
    └── ...

### AlphaPose Usage 

#### Inference
For inference with the alphapose folder as active directory use (example command):
<br>```python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video E:\alphapose\AlphaPose\examples\demo\DJI_0002.MP4 --save_video --vis_fast```

And the output will be in 

    .
    ├── ...
    ├── examples
    │   └── res
    │       ├── alphapose-results.json # Json with keypoints
    │       └── AlphaPose_DJI_0002.mp4 # Video
    └── ...

#### Finetuning

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

