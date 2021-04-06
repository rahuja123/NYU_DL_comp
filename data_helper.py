import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import  transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# The dataset class for unlabeled data.

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """
        self.split = split
        self.transform = transform
        self.basicTransform  =  transforms.Compose([
            transforms.ToTensor(),
           # normalize,
        ])
        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.split == 'unlabeled':
                label = self.basicTransform(img)
            else:
                label = self.labels[idx]
        return self.transform(img), label
    