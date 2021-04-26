# Feel free to modifiy this file.

from torchvision import models, transforms
import torch


team_id = 18
team_name = "JeRrY"
email_address = "ra3136@nyu.edu"

def get_model():
    model= models.resnet34(num_classes=800)
    return model

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
eval_transform = transforms.Compose([
    transforms.ToTensor(), normalize
])
