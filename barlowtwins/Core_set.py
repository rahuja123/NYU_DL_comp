import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image # PIL is a library to process images
from data_helper import CustomDataset
from sklearn.decomposition import PCA

import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
import sklearn.cluster as cluster

parser = argparse.ArgumentParser(description='Find Subset of IMages')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('pretrained', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('samples', type=Path, metavar='N',
                    help='samples wanted')
parser.add_argument('PCA', type=Path, metavar='N',
                    help='pca compression')
args = parser.parse_args()
train_transform = transforms.Compose([
        transforms.ToTensor(),  # convert PIL to Pytorch Tensor
        #normalize,
    ])
batch_size=256
trainset = CustomDataset(root=args.data, split="unlabeled", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=0)

dataset_length = len(trainset)
idxs_unlabeled = np.arange(dataset_length)

model = models.resnet50().cuda()
model.fc = nn.Identity()
state_dict = torch.load(args.pretrained, map_location='cpu')
model.load_state_dict(state_dict)

def get_embedding(Dataloader, model):
    embedding = torch.zeros([dataset_length, 2048])
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            x_in = batch[0].to(device=device)
            e1 = model(x_in)
            embedding[i*batch_size:(i+1)*batch_size] = e1.cpu()
            if(i%100)==0:
                print(i*batch_size/dataset_length)
    return embedding
train_transform = transforms.Compose([
        transforms.ToTensor(),  # convert PIL to Pytorch Tensor
    ])

embed = get_embedding(trainloader, model)
#embed = np.load('embed_info.npy')
# = embed[:100000]
pca = PCA(n_components=args.PCA)
pca.fit(embed)
compress = pca.transform(embed)

samples_desired = args.samples
embedding = compress#.numpy()
#The lb_flag is only useful for keeping track of which values are labeled
#If you can ask for data multiple times. As is it doesn't matter.
cluster_learner = cluster.KMeans(n_clusters=samples_desired)
#cluster_learner = cluster.MiniBatchKMeans(n_clusters=samples_desired,  max_iter=100, batch_size=100000, verbose=True)
cluster_learner.fit(embedding)
cluster_idxs = cluster_learner.predict(embedding)
centers = cluster_learner.cluster_centers_[cluster_idxs]
dis = (embedding - centers)**2
dis = dis.sum(axis=1)

q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(samples_desired)])

np.save('core_set_samples', q_idxs, allow_pickle=True, fix_imports=True)
