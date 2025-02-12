# Train file for MLP and LSTM
import yaml
import torch
#from easydict import EasyDict as edict
import easydict
from utils import update_config


def main():
    # Load config 
    cfg = update_config("config.yaml")
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    file_name = cfg.DATASET.FILE_PREFIX # Prefix for json files with annotated keypoints
    path = cfg.DATASET.ROOT_PATH # Path to folder with annotated jsons




if __name__ == '__main__':
    main()
