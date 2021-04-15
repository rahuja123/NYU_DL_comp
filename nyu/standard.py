#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning
#
# Implementation of a standard neural network (no self-supervised components).
# This is used as baseline (upper bound) and during linear-evaluation and fine-tuning.

import math
import time

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
import numpy as np

class StandardModel(torch.nn.Module):
    def __init__(self, feature_extractor, num_classes, tot_epochs=200):
        super(StandardModel, self).__init__()
        self.num_classes = num_classes
        self.tot_epochs = tot_epochs
        self.feature_extractor = feature_extractor
        feature_size = feature_extractor.feature_size
        self.classifier = nn.Linear(feature_size, num_classes)

    def forward(self, x, detach=False):
        if(detach): out = self.feature_extractor(x).detach()
        else: out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.feature_extractor.load_state_dict(checkpoint["backbone"])
