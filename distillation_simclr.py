#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time
from data_helper import CustomDataset
import torchvision.models as models
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

start_time = time.time()

batch_size = 1
test_batch_size = 1
epochs = 1
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 1
log_interval = 10

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}

transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

rainset = CustomDataset(root = 'dataset/',split = 'unlabeled', transform = transformer)
evalset = CustomDataset(root = 'dataset/',split = 'train', transform = transformer)
testset = CustomDataset(root = 'dataset/',split = 'val', transform = transformer)

# subset_percent = 0.001 #change it to 1 for full dataset
# trainset_size = len(trainset)
# indices = list(range(trainset_size))
# split = int(np.floor(subset_percent * trainset_size))
# np.random.seed(230)
# np.random.shuffle(indices)
# train_sampler = SubsetRandomSampler(indices[:split])
train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,sampler=train_sampler,shuffle=False, **kwargs)
eval_loader = torch.utils.data.DataLoader(evalset,batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset,batch_size=test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self,num_channels):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = num_channels

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)
        self.dropout_rate = 0.0

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4*4*self.num_channels*4)             # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 10

        return s

teacher_model = models.resnet50(pretrained=False,num_classes=800)
teacher_model = teacher_model.cuda() if cuda else teacher_model
teacher_model = torch.load('resnet50.pth')

model = models.resnet18(pretrained=False,num_classes=800).cuda() if cuda else models.resnet18(pretrained=False,num_classes=800)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def distill_unlabeled(y, teacher_scores, T):
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * T * T

def train(epoch, model, loss_fn):
    model.train()
    teacher_model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach()
        # teacher_output = Variable(teacher_output.data, requires_grad=False) #alternative approach to load teacher_output
        loss = loss_fn(output, teacher_output, T=10.0)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def train_evaluate(model):
    model.eval()
    train_loss = 0
    correct = 0
    for data, target in eval_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
#         print(target)
        train_loss += F.cross_entropy(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, epochs + 1):
    train(epoch, model, loss_fn=distill_unlabeled)
    train_evaluate(model)
    test(model)
