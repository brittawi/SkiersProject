# AI and skiing (Machine Learning-Enhanced Ski Trainer Based on Expert Movement Analysis) 

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Keypoint annotations](#keypoint-annotations)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)

## Project Description (Britta)
**TODO**

Our goal is to provide **technique feedback for cross-country skiers** using machine learning. While previous research has primarily focused on **classifying sub-techniques**, little work has been done on **providing (real-time) feedback** based on movement patterns. Most studies have relied on **sensor-based classification** ([[1]](#1), [[3]](#3), [[4]](#4)), while video-based classification has been limited to **controlled treadmill environments**—where sensors have shown better performance ([[2]](#2)).

In this project, we aim to **classify cross-country skiing sub-techniques using non-static drone video data** and **provide feedback to skiers** based on their movement.

**Scope of the Project**</br>
Cross-country skiing consists of two main styles: **classic** and **skating**. This project focuses on the **skating style**, specifically **gear two and gear three**, as these sub-techniques are available in our dataset.

As part of the project we want to:

1. **Train a pose estimation model** to detect joint key points automatically.
2. **Use these key points to classify** skating sub-techniques (gear two & three) and their cycle phases.
3. **Apply Long Short-Term Memory (LSTM) networks** for classification, as they have been effective in previous studies ([[1]](#1), [[2]](#2), [[4]](#4)). We also want to test a simple **MLP network**. 
4. **Use Dynamic Time Warping (DTW)** to compare user movement against expert data, generating technique feedback.

By leveraging machine learning and pose estimation, this project aims to enhance technique analysis in cross-country skiing, providing athletes with meaningful, data-driven feedback.

TODO put Halpe dataset here as well?!

## Dataset
**TODO**
The dataset that will be used for this project is drone captured videos of an expert skier skating on flat ground. The skating techniques used in the videos consist mainly of gear two and three. The skier is captured from several different viewpoints but the two we will use are from the front and from the sides based on an interview with a ski coach. Videos in different conditions (e.g. snow, darkness, etc.) and with various levels of skiers might be added later in the project.  

**TODO**
Talk about annotated data?!
Talk about data that we will add

## Keypoint annotations
**TODO**
Talk about annotated data?!
Talk about Halpe dataset?!

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
|CHOOSEN_JOINT | With this parameter we can set a single joint that we want to use for detecting a cycle. It is only possible to set one joint for this. |
|CHOOSEN_REF| Sets a reference joint to compute relative movements of other joints. By doing so we can better compare the cycles. We have used the Hip as a reference joint. |
|CHOOSEN_DIM| Determines the dimension (x or y) to be used for cycle detection. |
|sigma_value| Defines the smoothing factor for the signals. We are using Gaussian smoothing. |
|order| Sets the minimum spacing between detected local minima in the X and Y movement data, filtering out minor fluctuations. (Is this right??) |
|LABELS| This parameter contains a dictionary with possible labels and keys that can be used as some kind of shortcut for labeling (more details further down). If more gears should be added, this has to be done here. |
|CHOOSEN_KEYPOINTS| Sets the joints that should be saved for each cycle. Both x and y value will be saved to a json file by default for each joint.|
|VIDEO_DIR| Path to directory where the video files are stored. |
|ANNO_DIR| Path to the directory where the corresponding annotations in COCO format are stored. |
|video_id| ID of video which should be labeled. |

When running the Jupyter notebook a new window will be opened where your selected video is being played. It will pause after the first cycle. You can then enter the following options: 
- "2" : Gear 2
- "3" : Gear 3
- "4" : Gear 4
- "u" : Unknown (Transition or if no gear can be identified)
- "r" : Replay the cycle
- "q" : Quit the program

As **output** a json file will be saved in your working directory. The json file will have the following structure:
```
{
    "Cycle 1" : {
        "RAnkle_x" : [...],
        "RAnkle_y : [...],
        "Label": "gear2",
        "Start_frame": 34,
        "End_frame": 100
    },
    "Cycle 2" : {
        "RAnkle_x" : [...],
        "RAnkle_y : [...],
        "Label": "gear3",
        "Start_frame": 100,
        "End_frame": 173
    },
    ...
}
```
Each Cycle contains the keypoints for the choosen joints in that time period, a label and the start and end frame. 

#### Training (Emil)

#### Create Annotations (Emil)

#### Other models (Britta)
(write about Installation of MMPose?)

We have tested the following pose estimation models for this projects (TODO):

||**MediaPipe**|**YoloNas**|**MMPose**|**OpenPose**|**AlphaPose (chosen)**|
|-|------------|-----------|----------|------------|-------------|
|**Images detection**|11 sec|6 min|37 sec|||
|**Video detection**|2 min?|Over 1 h|6 min|88 sec|44 sec|
|**Issues**|Keypoint annotations were very bad|Good annotations but very slow|Very good annotations but feet are not detected|Feet are annotated, but annotations were quite jumpy|Annotations are less accurate than with MMPose but still very good and feet are also annotated|

**Criteria for choosing the pose estimation model:**
- Accuracy of keypoint annotations
- Duration for detection
- Predicted keypoints (We wanted to include all main joints of the body. After meeting with a crosscountry ski trainer we looked for a model that also includes keypoints on the foot.)




## Contribution
TODO not sure if we even need this  

## TODO
- [ ] Feedback system
- [ ] Crossvalidation


## References
<a id="1">[1]</a> 
Gupta, S. S., Johansson, M., Kuylenstierna, D., Larsson, D., Ortheden, J., & Pettersson, M. (2022). Machine learning techniques for GAIT analysis in skiing. In Advances in intelligent systems and computing (pp. 126–129). https://doi.org/10.1007/978-3-030-99333-7_21 

<a id="2">[2]</a> 
Nordmo, TA.S., Riegler, M.A., Johansen, H.D., Johansen, D. (2023). Arctic HARE: A Machine Learning-Based System for Performance Analysis of Cross-Country Skiers. In: Dang-Nguyen, DT., et al. MultiMedia Modeling. MMM 2023. Lecture Notes in Computer Science, vol 13833. Springer, Cham. https://doi.org/10.1007/978-3-031-27077-2_43 

<a id="3">[3]</a> 
Pousibet-Garrido, A., Polo-Rodríguez, A., Moreno-Pérez, J. A., Ruiz-García, I., Escobedo, P., López-Ruiz, N., Marcen-Cinca, N., Medina-Quero, J., & Carvajal, M. Á. (2024). Gear Classification in Skating Cross-Country Skiing Using Inertial Sensors and Deep Learning. Sensors, 24(19), 6422. https://doi.org/10.3390/s24196422 

<a id="4">[4]</a> 
Rassem, A., El-Beltagy, M., & Saleh, M. (2017). Cross-Country Skiing Gears Classification using Deep Learning. arXiv:1706.08924. https://doi.org/10.48550/arXiv.1706.08924 

<a id="5">[5]</a> 
Wang, J., Qiu, K., Peng, H., Fu, J., & Zhu, J. (2019). AI Coach: Deep human pose estimation and analysis for personalized athletic training assistance. Proceedings of the 30th ACM International Conference on Multimedia. https://doi.org/10.1145/3343031.3350910 

<a id="6">[6]</a> 
Yue, C. Z., Yong, L. C., & Shyan, L. N. (2022). Exercise quality analysis using AI model and computer vision. Journal of Engineering Science and Technology Special Issue (pp. 157 - 171). https://jestec.taylors.edu.my/Special%20Issue%20SIET2022/SIET2022_10.pdf 

