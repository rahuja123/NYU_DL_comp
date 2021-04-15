import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from tqdm.notebook import tqdm
import torch.optim as optim
from utils.data_helper import CustomDataset
import VAE
from PIL import Image
from byol_pytorch import BYOL

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
args = parser.parse_args()
os.makedirs(args.checkpoint_dir, exist_ok=True)

train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.functional.to_grayscale()
        #transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
        #transforms.RandomRotation(20, resample=Image.BILINEAR),
        #transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
        transforms.ToTensor(),  # convert PIL to Pytorch Tensor
        #normalize,
    ])

trainset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transform)
trainset, validset = torch.utils.data.random_split(trainset, [100000, len(trainset)-100000], generator=torch.Generator().manual_seed(42))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

epochs = 100
lr = 1e-2
momentum=0.9

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_mse_loss_record = []
epochs = 1
momentum=0.9
save_path = "./model_save/"

resnet = models.resnet50(pretrained=False).to(device=device)
learner = BYOL(
    resnet,
    image_size = 96,
    hidden_layer = 'avgpool',
    projection_hidden_size = 4096,   
    moving_average_decay = 0.99,      
).to(device=device)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

for epoch in range(epochs):
    total_train_loss = 0.0
    for i, batch in enumerate(trainloader):
        x_in = batch[0].to(device=device)
        loss = learner(x_in)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        total_train_loss += loss.item()
    mean_train_loss = total_train_loss*16 / len(trainset)
    print(mean_train_loss)
    torch.save(resnet.state_dict(), os.path.join(args.checkpoint_dir, 'unlabel{}.pth'.format(epoch + 1)))
    #torch.save(model.state_dict(),save_path + 'unlabel{}.pth'.format(epoch + 1))

