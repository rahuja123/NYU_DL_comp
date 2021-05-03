# Feel free to modifiy this file.

from torchvision import models, transforms
import torch


team_id = 18
team_name = "JeRrY"
email_address = "ra3136@nyu.edu"

def get_model():
    model= models.resnet50(num_classes=800)
    return model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
