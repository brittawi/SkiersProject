# AI and skiing (Machine Learning-Enhanced Ski Trainer Based on Expert Movement Analysis) 

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Keypoint annotations](#keypoint-annotations)
- [AlphaPose installation and usage](#installation)
- [Gear Classification](#gear-classification)
- [Contribution](#contribution)

## Project Description (Britta)

Our goal is to provide **technique feedback for cross-country skiers** using machine learning. While previous research has primarily focused on **classifying sub-techniques**, little work has been done on **providing (real-time) feedback** based on movement patterns. Most studies have relied on **sensor-based classification** ([[1]](#1), [[3]](#3), [[4]](#4)), while video-based classification has been limited to **controlled treadmill environments**—where sensors have shown better performance ([[2]](#2)).

In this project, we aim to **classify cross-country skiing sub-techniques using non-static drone video data** and **provide feedback to skiers** based on their movement.

![project_overview](project_overview.png)
*Project overview*

**Scope of the Project**</br>
Cross-country skiing consists of two main styles: **classic** and **skating**. This project focuses on the **skating style**, specifically **gear two and gear three**, as these sub-techniques are available in our dataset.

As part of the project we want to:

1. **Annotate** selected videos.
2. **Finetune a pose estimation model** to detect joint key points automatically.
3. **Use these key points to classify** skating sub-techniques (gear two & three) and their cycle phases.
4. **Apply Long Short-Term Memory (LSTM) networks** for classification, as they have been effective in previous studies ([[1]](#1), [[2]](#2), [[4]](#4)). We also want to test a simple **MLP network**. 
5. **Use Dynamic Time Warping (DTW)** to compare user movement against expert data, generating technique feedback.

By leveraging machine learning and pose estimation, this project aims to enhance technique analysis in cross-country skiing, providing athletes with meaningful, data-driven feedback.

## Dataset
The dataset used in this project contains 149 drone-recorded videos of 12 cross-country skiers, captured using a DJI Mini 2 drone from front and side perspectives. The recordings were taken at Ormberget in Luleå, Sweden, under diverse natural conditions including sunshine, fog, and wind. Skiers of varying skill levels—from beginners to national-level athletes—performed skiing techniques in gears G2, G3, and G4, as well as sprints and transitions. Expert skiers also simulated five common technique mistakes to support automated feedback development. Manual drone tracking ensured dynamic yet informative footage suitable for pose estimation and motion analysis.
From these videos we have created a dataset both for fine-tuning the pose estimation model and for training a classifier to detect different gears.

### Dataset for fine-tuning the pose estimation model
A subset of the data (18 full videos and 10 short clips) was manually annotated using CVAT, yielding 13,977 annotated frames across 10 different skiers. Further information can be found under: (include link to dataset).

The folder create_annotations includes a file (split_annotated_videos) to split annotated videos into a train, validation and test folder so that they can be used for fine-tuning a pose estimation model. For this the annotated data needs to be in COCO format. The second file(getAnnotationsFromAlphaPose) can be used to convert annotations from the halpe to the coco format. When running AlphaPose on a video the output will be in the halpe format.

### Dataset for training a classifier
To train a classifier we have used the fine-tuned AlphaPose model to estimate the keypoints on all the recorded videos. This can be done by using the file test_pipe.py under the folder feedback_system. Using this file will output annotations per video. Next we have labeled the cycles accordingly. This can be done with the script lable_cycles.ipynb under the folder classification/classify_splits. Again the annotated data has to be in COCO format. This process is further explained under [this section](#label-cycles-britta). Our labeled data can be found under data/labeled_data.

## Keypoint annotations
We have chosen AlphaPose as the pose estimation model. This is using the [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody) 26 keypoints: 
```
    //26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"}
```

## AlphaPose installation and usage

This section explains the installation process and a breif usage guide for finetuning AlphaPose. In the feedback_system AlphaPose is implemneted and used automatically.  

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

To use pretrained Halpe26 model download from the [Model Zoo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md) and put the halpe26_fat_res50_256x192.pth file into the pretrained_models folder. Also our pretrained models from the OneDrive should be put here. Note put it in the `/pretrained_models` folder under `/alphapose` and not the `/pretrained_models` higher as that is for gear classification:

    .
    ├── alphapose
    │   ├── ...
    │   ├── pretrained_models
    │   │   └── halpe26_fat_res50_256x192.pth
    │   └── ...
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

2. Put json from cvat into a folder to load from and the same as the video id for example "02.json". Set annotation_folder to the path to this folder in split_video_to_jpg.ipynb and optionally change combined_json_path for output file. 

3. Put split images and annotation file into a training folder to load from. Adopted the [AlphaPose install.md](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md) our file structure is:
```
    .
    ├── alphapose
    │    ├── alphapose
    │    ├── .tensorboard
    │    ├── configs
    │    ├── data
    │    │   └── halpe 
    │    │       ├── annotations
    │    │       │   ├── halpe_train_v1.json
    │    │       │   └── halpe_val_v1.json
    │    │       ├── images
    │    │       └── ├── train2015
    │    │           │   ├── HICO_train2015_00000001.jpg
    │    │           │   ├── HICO_train2015_00000002.jpg
    │    │           │   ├── HICO_train2015_00000003.jpg
    │    │           │   ├── ... 
    │    │           └── val2017
    │    │               ├── 000000000139.jpg
    │    │               ├── 000000000285.jpg
    │    │               ├── 000000000632.jpg
    │    │               ├── ...
    │    ├── ...
    ├── classification
    ├──...
```
So now put the images and json into this structure or change it in the .yaml in upcomming steps.

5. Open and edit config in:
```
    .
    ├── exp
    ├── alphapose
    ├── configs
    │   └── 256x192_res50_lr1e-3_1x.yaml
    ├── ...
```
Here you set the train and validation image load folder, annotation files, and set which pretrained weights to use. To use pretrained weights set:
```PRETRAINED: 'pretrained_models/halpe_26_fast_res50_256x192.pth'``` Also change other configs here such as learning rate, epochs, etc. 

6. Run training command:
```python scripts/train.py --exp-id trained_models --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml```
```--exp_id``` sets the output folder and the trained weights can be found in /exp, for example:

```
    .
    ├── ...
    ├── exp
        └── trained_models-256x192_res50_lr1e-3_1x.yaml
            ├──
    ├── ...
```

7. Run inference with new trained weights, put the trained weights into `/pretrained_models` folder (under alphapose and not in main) and in the inference command change ```--checkpoint``` to the weight file name, for example

```python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/final_DPG_iter_2.pth --video E:\alphapose\AlphaPose\examples\demo\DJI_0015.MP4 --save_video --vis_fast```

### Testing

```python scripts/test.py --cfg configs/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/RegLoss100EpochNoFlipDPG.pth --num_workers 4```

## Gear Classification

Under ```/classification``` are most files related to gear classification and related material. In ```/cycle_splits``` are notebooks related to investigating the joint singals (`cycle_plots_2D.ipynb`, `cycle_plots_front.ipynb`, and `front_test.ipynb`), and the notebook used for labeling cycles `label_cycles.ipynb`.  The ```/training``` folder contains most files for setting up the data, optimizing hyperparameters, and trainng and testing the models for gear classification. Finally in `/classify_angle` is just a single test file for testing the viewangle classification method and in `/cycles_stats` a single notebook for creating plots about the dataset.

### Label Cycles
With the file `label_cycles.ipynb` a video of a crosscountry skier can be split into cycles and the cycles can be labeled with the corresponding gear. To do so the video and the keypoint annotations are needed. The keypoint annotations need to follow the COCO format and therefore contain the following section:

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

Under Config in the notebook you can modify the following parameters:
|Param  | Description   |
|------ |---------------|
|CHOOSEN_JOINT | With this parameter we can set a single joint that we want to use for detecting a cycle. It is only possible to set one joint for this. At the moment this parameter is not used. The chosen joint is instead selected based on the view angle.|
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

### Training

The ```/training``` folder contains five main runnable files for setting up the data, conducting hyperparameter optimization, cross-validation, training final models, and testing models for gear classification of the cycles. It also contains train and test results of various runs and controlling yaml files. 

First, the file ```setup_data.py``` is used for combining the label cycle json files and splitting into train and test sets (in here the config is not used so change values in the file if desired). The parameters in ```setup_data.py``` are:

| Param             | Description                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| IN_PATH         | Relative path to the directory containing the original labeled `.json` data files from the `label_cycles.ipynb` output.     |
| OUT_PATH       | Relative path to the directory where the split train/test `.json` files will be saved. |
| TRAIN_FILE_NAME | Name of the output file that will store the training dataset after splitting.          |
| TEST_FILE_NAME  | Name of the output file that will store the test dataset after splitting.              |
| test_size         | Proportion of the dataset to be used as the test set during the train-test split.                                                         |
| seed             | Random seed value used to ensure reproducibility of the train-test split.                                                                 |
| EXLUDED_SKIER_IDS | List of skier IDs that should be excluded from the dataset before splitting. Used to filter out specific individuals to create alternate dataset. |



The file ```tune_hyperparameters.py``` load a search space parameters from ```search_space.yaml``` and other run parameters from ```config.yaml``` and use raytune to optimize models and output the best found paramateres in ```/ray_tune```. However, make sure it is the same network type and number of epochs and max epochs in ```config.yaml``` and ```search_space.yaml```. The file ```train_cv.py``` does cross validation with each fold getting the average from the seeds listed in the config. In ```/run``` it outputs plots of the averaged metrics in ```/plots```, each model from each seed in each fold in ```/saved_models```, and the tensorboard logs in ```/tensorboard_runs```. Run ```train_final.py``` to train a final classification model by manually inserting the best found hyperparameters from hyperparameter optimization and train for the number of epochs with lowest average validation loss from cross-validation. It will train for the numbers of epochs given in the config TRAIN.EPOCHS. Finally, ```test.py``` will run the final test on the hold-out test set created from ```setup_data.py```. This file does not load from the config file but the parameters are saved together with the model weights during training and loads from there, so to set the path to the model weight and test data you have to change `MODEL_PATH` and `TEST_DATA_PATH` in the file. All the LSTM and mlp folders are renamed runs folders contining the specified results.

To train and use your own models for the feedback system take any of the resulting `.pth` model weights and put into `/pretrained_models` and then change the `CLS_GEAR.MODEL_PATH` parameter in `pipe_test.yaml` under `/feedback_system` to the weight name.  


Parameters in ```config.yaml```:

**DATASET**
| Param            | Description |
|-----------------|-------------|
| TRAIN_SIZE | Proportion of the dataset used for training in hyperparameter optimization. |
| VAL_SIZE | Proportion of the dataset used for validation in hyperparameter optimization. |
| ROOT_PATH | Relative path to the dataset. |
| ROOT_ABSOLUTE_PATH | Absolute path to the dataset for reference. Used in hyperparameter optimization because raytune have different root.  |
| AUG: SMOOTHING | Smoothing factor applied to the dataset. |
| AUG: NORMALIZATION | Determines if dataset normalization should be applied. |
| AUG: NORM_TYPE | Type of normalization applied to the dataset `full_signal` or `per_timestamp`. |

**DATA_PRESET**
| Param            | Description |
|-----------------|-------------|
| CHOOSEN_JOINTS | List of selected joints used for classification, use both x and y value e.g. `LShoulder_x` and `LShoulder_y`. |
| LABELS | Dictionary mapping cycle labels to numeric values. |

**TRAIN**
| Param            | Description |
|-----------------|-------------|
| BATCH_SIZE | Number of samples per batch during training. |
| EPOCHS | Total number of training epochs. |
| OPTIMIZER | Optimization algorithm used. |
| LOSS | Loss function used `cross_entropy` or `focal_loss`. |
| LR | Learning rate for the optimizer. |
| PATIENCE | Number of epochs with no improvement before early stopping (implemented but not currently used). |
| K_FOLDS | Number of folds used for k-fold cross-validation. |
| SEEDS | List of random seeds for reproducibility. The first seed in this list is used for the cross validation split and seed in train/val split for hyperparemeter optimization.  |

**TRAIN.NETWORK**
| Param            | Description |
|-----------------|-------------|
| NETWORKTYPE | Specifies the neural network architecture `lstm` or `mlp`. |
| LSTM: HIDDEN_SIZE | Number of hidden units in the LSTM network. |
| LSTM: NUM_LAYERS | Number of stacked LSTM layers. |
| LSTM: DROPOUT | Dropout rate applied to the LSTM layers. |
| MLP: HIDDEN_1 | Number of neurons in the first hidden layer of MLP. |
| MLP: HIDDEN_2 | Number of neurons in the second hidden layer of MLP. |

**LOGGING**
| Param            | Description |
|-----------------|-------------|
| ROOT_PATH | Root path for all the outputs. |
| TENSORBOARD_PATH | Path for TensorBoard logs within `ROOT_PATH`. |
| MODEL_DIR | Path for saved models within `ROOT_PATH`. |
| PLOT_PATH | Path for plots within `ROOT_PATH`. |
| BEST_EPOCH_PATH | Path for txt file with stats about lowest loss epoch in corss-validation within `ROOT_PATH`. |

### OPTIMIZATION
| Param            | Description |
|-----------------|-------------|
| SEARCH_CONFIG | .yaml file defining the hyperparameter search space. |
| OUTPUT_ROOT | Root directory for saving training outputs and checkpoints. |
| CHECKPOINTS: ENABLE | Enables checkpointing during training `true` or `false`. |

#### Other models (Britta)
We have tested the following pose estimation models for this projects:

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

