import torch
from torch.utils.data import Dataset
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, 
                 data,
                 labels, 
                 label_dict = {
                    "unknown": 0,
                    "gear2" : 1,
                    "gear3" : 2,
                    "gear4" : 3,}
                 ):
        
        self.data = data
        self.labels = labels
        self.label_dict = label_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.label_dict[self.labels[idx]]
        # Check if tensor and only convert if so, recommended by user warning
        if isinstance(self.data[idx], torch.Tensor):
            item = self.data[idx].to(torch.float32) # Just assuring correct type
        else:
            item = torch.tensor(self.data[idx], dtype=torch.float32)
        item = self.data[idx].T
        
        return item, label