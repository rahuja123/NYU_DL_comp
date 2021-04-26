# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.optim import lr_scheduler
from .WarmupMultiStepLR import WarmupMultiStepLR


def make_scheduler(sched,optimizer):

    if sched= "WarmupMultiStepLR":
        scheduler = WarmupMultiStepLR(optimizer, [30, 55], 0.1, 0.01, 10, "linear")
    else:
        scheduler = lr_scheduler.StepLR(optimizer, 40, 0.1)
    return scheduler
