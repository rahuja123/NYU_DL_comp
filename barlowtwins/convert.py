import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default=None,type=str)
args = parser.parse_args()


net = models.resnet50(num_classes=800)
ckpt= torch.load(args.checkpoint_path)
msg= net.load_state_dict(ckpt['model'], strict=False)
print(msg.missing_keys)

classifier = torch.nn.Linear(2048, 800)
classifier.weight.data.normal_(mean=0.0, std=0.01)
classifier.bias.data.zero_()

state_dict= ckpt['classifier']
for k in list(state_dict.keys()):
  if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
  del state_dict[k]

classifier.load_state_dict(state_dict)
net.fc= classifier

torch.save(net.state_dict(), 'barlow_nyu_original.pth')
