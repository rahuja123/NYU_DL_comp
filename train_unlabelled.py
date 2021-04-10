import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm.notebook import tqdm
import torch.optim as optim
from utils.data_helper import CustomDataset
import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
args = parser.parse_args()

# These numbers are mean and std values for channels of natural images. 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.functional.to_grayscale()
        transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
        transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
        transforms.ToTensor(),  # convert PIL to Pytorch Tensor
        #normalize,
    ])

trainset = CustomDataset(root='/dataset', split="unlabelled", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

model = VAE.VAE(1600)
epochs = 100
lr = 1e-2
momentum=0.9

def loss_function(xhat, x, mu, logvar, kl_weight=1.0):
    BCE = nn.functional.mse_loss(xhat, x, reduction='none').sum(1).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1).mean()

    loss = BCE + kl_weight*KLD
    return loss, BCE, KLD

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.MSELoss()

model.train()
if torch.cuda.is_available():
    model = model.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)


train_mse_loss_record = []

for epoch in range(epochs):
    total_train_loss = 0.0
    count = 0;
    for batch in tqdm(trainloader, leave=False):
        x_in = batch[0].to(device=device)
        x_label = batch[1].to(device=device)
        outputs, mu, logvar = model(x_in)
        #loss, bce, kld = loss_function(outputs, x_label, mu, logvar, 0.5)
        loss = criterion(outputs,x_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        count+=1
        if count %10==0:
            print(total_train_loss/10)
            total_train_loss=0.0

    mean_train_loss = total_train_loss / len(trainset)

    train_mse_loss_record.append(mean_train_loss)
    print(mean_train_loss)
    if epoch%10 ==0 :
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'unlabel{}.pth'.format(epoch + 1)))
        #torch.save(model.state_dict(),save_path + 'unlabel{}.pth'.format(epoch + 1))
print('Finished Training')

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_demo.pth')}")
