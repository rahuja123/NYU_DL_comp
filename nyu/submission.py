# Feel free to modifiy this file.

from torchvision import models, transforms
from standard import StandardModel
import torch
from resnet_large import ResNet, BasicBlock


team_id = 18
team_name = "JeRrY"
email_address = "ra3136@nyu.edu"

def get_model():
    resnet34 = ResNet(BasicBlock, layers=[3, 4, 6, 3],zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None)
    # feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    model = StandardModel(resnet34, 800)
    return model

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
eval_transform = transforms.Compose([
    transforms.ToTensor(), normalize
])
