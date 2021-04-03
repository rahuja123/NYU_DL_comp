import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from utils.data_helper import CustomDataset
from model import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
args = parser.parse_args()

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = CustomDataset(root='/dataset', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

evalset = CustomDataset(root='/dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)


if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

net = VAE().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

print('Start Training')
net.train()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            
    if epoch%20==0:
        eval()

print('Finished Training')

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "net_demo.pth"))

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'net_demo.pth')}")

def eval():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in evalloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f" Accuracy: {(100 * correct / total):.2f}%")