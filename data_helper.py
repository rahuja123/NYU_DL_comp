import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import  transforms


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


class UpdatedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """
        split = 'unlabeled'
        self.transform = transform
        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, "label_18.pt")
        idx_path = os.path.join(root, "request_18.csv")
        self.num_images = 12800
        self.ids = np.loadtxt(idx_path, delimiter=",")  
        self.ids = self.ids.astype(int)
        self.labels = torch.load(label_path)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{self.ids[idx]}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')
            label = self.labels[idx]
        return self.transform(img), label
