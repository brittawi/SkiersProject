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

And the output will be in:

    .
    ├── ...
    ├── examples
    │   └── res
    │       ├── alphapose-results.json # Json with keypoints
    │       └── AlphaPose_DJI_0002.mp4 # Video
    └── ...

Then just convert the output format to coco using getAnnotationsFromAlphaPose.ipynb

#### Finetuning

1. Convert videos to images. Put videos for finetuning into a folder and create output folder for split_video_to_jpg.ipynb to split the videos into images. Set video_folder path to the input folder and output_frame_path to output folder in split_video_to_jpg.ipynb. 

2. Put json from cvat into a folder to load from and name as {video} + "_annotations.json" for example "DJI_0002_annotations.json". Set annotation_folder to the path to this folder in split_video_to_jpg.ipynb and optionally change combined_json_path for output file. 

3. Put split images and annotation file into a training folder to load from. From the [AlphaPose install.md](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md) the file structure is:
```
    .
    ├── json
    ├── exp
    ├── alphapose
    ├── configs
    ├── test
    ├── data
    └── ├── halpe
        └── ├── annotations
            │   ├── halpe_train_v1.json
            │   └── halpe_val_v1.json
            ├── images
            └── ├── train2015
                │   ├── HICO_train2015_00000001.jpg
                │   ├── HICO_train2015_00000002.jpg
                │   ├── HICO_train2015_00000003.jpg
                │   ├── ... 
                └── val2017
                    ├── 000000000139.jpg
                    ├── 000000000285.jpg
                    ├── 000000000632.jpg
                    ├── ...
```
So now put the images and json into this structure or change it in the .yaml in upcomming steps. We do not use the validation set so that folder and json is not needed.

5. Open and edit config in:
```
    .
    ├── json
    ├── exp
    ├── alphapose
    ├── configs
    │   ├── ...
    │   └── halpe_26
    │       └── resnet
    │           ├── 256x192_res50_lr1e-3_1x.yaml # This one 
    │           └── ...
    ├── ...
```
Here you set the train image load folder, annotation file, and set which pretrained weights to use. To use pretrained weights set:
```PRETRAINED: 'pretrained_models/halpe_26_fast_res50_256x192.pth'``` Also change other configs here such as learning rate, epochs, etc. 

6. Run training command:
```python scripts/train.py --exp-id trained_models --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml```
```--exp_id``` sets the output folder and the trained weights can be found in /exp, for example:

```
    .
    ├── ...
    ├── exp
        └── ...
    ├── ...
```

7. inference command (change xx to video nr in --video, change --checkpoint to trained weights, move videos to folder) 

python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/final_DPG_iter_2.pth --video E:\alphapose\AlphaPose\examples\demo\DJI_0015.MP4 --save_video --vis_fast 

8. convert output json to coco format 

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

