import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from data_helper import CustomDataset


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
dataset = CustomDataset(root='dataset/',split='unlabeled',transform=transform)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train = CustomDataset(root = 'dataset/', split = 'train', transform = transform)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        corrects = pred.eq(target.view(1, -1).expand_as(pred))
        correct = 0.0
        for i in corrects[0]:
            if i.item() == True:
                correct+=1
        return (correct/float(len(target)))*100




class classifier_class(nn.Module):

    def __init__(self, nclasses):
        super(classifier_class, self).__init__()
        model= models.resnet34(num_classes=800)
        model.load_state_dict(torch.load('model1.pth'))
        self.encoder = torch.nn.Sequential(*(list(model.children())[:-1]))# models.resnet34(pretrained=False).to(device=device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.linear1= nn.Sequential(nn.Linear(512,512), nn.LeakyReLU())
        self.linear2= nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512,nclasses),
        )
    def forward(self, x):
        mu = self.encoder(x)
        out = F.leaky_relu(mu)
        # print(out.shape)
        out= out.view(out.shape[0], out.shape[1])
        out = self.linear1(out)
        out = self.linear2(out)
        return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            #if isinstance(param, self.Parameter):
            else:
                # backwards compatibility for serialized parameters
                param = param.data
            ownstate[name].copy(param)





val = CustomDataset(root = 'dataset/', split = 'val', transform = transform)
val_loader = DataLoader(val, batch_size=128, num_workers=0, drop_last=False, shuffle=False)
train_loader = DataLoader(train, batch_size=128, num_workers=0, drop_last=False, shuffle=True)
feature_extract=True
model_ft_2 = classifier_class(800)
# model_ft_2 = models.resnet18(num_classes=800, pretrained=False)
# set_parameter_requires_grad(model_ft_2, feature_extract=True)
# num_ftrs_2 = model_ft_2.fc.in_features
# model_ft_2.fc = nn.Linear(num_ftrs_2, 800)
input_size = 96
criterion_2 = torch.nn.CrossEntropyLoss()

model_ft_2.cuda()
criterion_2.cuda()

print(model_ft_2)


params_to_update = model_ft_2.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft_2.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
#             /print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

from tqdm import tqdm
epochs = 120
best_acc=0
epoch = 0
for epoch in range(epochs):
    top1= 0
    top1_val=0
    for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
        x_batch= x_batch.cuda()
        y_batch= y_batch.cuda()
        logits = model_ft_2(x_batch)
        loss = criterion_2(logits, y_batch)
        top1 += accuracy(logits, y_batch, topk=(1,))
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()
    top1_accuracy = top1/(counter + 1)
    print("loss= ", loss)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_accuracy}")
    for counter, (x_batch, y_batch) in enumerate(tqdm(val_loader)):
        x_batch= x_batch.cuda()
        y_batch= y_batch.cuda()
        logits = model_ft_2(x_batch)
#         loss = criterion_2(logits, y_batch)
        top1_val += accuracy(logits, y_batch, topk=(1,))
    top1_accuracy_val = top1_val/(counter + 1)
    print(f"Epoch {epoch}\tTop1 Val accuracy {top1_accuracy_val}")

torch.save(model_ft_2.state_dict(), 'r34_dropout.pth')

def predict_prob_dropout_split(X,n_drop):
#         loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
#                             shuffle=False, **self.args['loader_te_args'])

#         model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        loader_te = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False, shuffle=True)
        probs = torch.zeros([n_drop, len(X), 800])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                idxs = 0
                for x, y in loader_te:
                    x= x.cuda()
                    y= y.cuda()
#                     x, y = x.to(self.device), y.to(self.device)
                    out = model_ft_2(x)
                    # print(out.shape)
#                     print(out.shape)
                    probs[i][idxs] += F.softmax(out[0], dim=0).cpu()
                    idxs+=1
        return probs

def query(n):
    n_pool = 512000
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_unlabeled = np.arange(n_pool)[~idxs_lb]
#     print(idxs_unlabeled.shape)
    probs = predict_prob_dropout_split(dataset, 10)
    pb = probs.mean(0)
    entropy1 = (-pb*torch.log(pb)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    U = entropy2 - entropy1
    return idxs_unlabeled[U.sort()[1][:n]]


X = query(12800)
print(X)
print(type(X))


import csv
with open('bald.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(X)):
        writer.writerow(str([X[i]])+',')
