import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from  data_helper import CustomDataset
from submission import get_model, eval_transform, team_id, team_name, email_address

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default='/home/rahulahuja/nyu/dl/NYU_DL_comp/moco/checkpoint_96_800.pth.tar',type=str)
args = parser.parse_args()

evalset = CustomDataset(root='../dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)


net = get_model()
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

torch.save(net.state_dict(), 'barlow_nyu_extra.pth')





# state_dict = torch.load('/home/rahulahuja/nyu/dl/NYU_DL_comp/barlowtwins/checkpoint/resnet50.pth', map_location='cpu')
# net.load_state_dict(state_dict, strict=False)
#
# classifier = torch.nn.Linear(2048, 800)
# classifier.weight.data.normal_(mean=0.0, std=0.01)
# classifier.bias.data.zero_()
#
# ckpt= torch.load('/home/rahulahuja/nyu/dl/NYU_DL_comp/barlowtwins/checkpoint/lincls/checkpoint.pth', map_location='cpu')
# state_dict= ckpt['classifier']
# for k in list(state_dict.keys()):
#   if k.startswith('module.'):
#       state_dict[k[len("module."):]] = state_dict[k]
#   del state_dict[k]
#
# classifier.load_state_dict(state_dict)
# net.fc= classifier
# net.load_state_dict(torch.load(args.checkpoint_path))
# # net.load(args.checkpoint_path)
# checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
# state_dict= checkpoint['state_dict']
# print(state_dict.keys())
# # checkpoint = torch.load(args.pretrained, map_location="cpu")
# # state_dict = checkpoint['state_dict']
# for k in list(state_dict.keys()):
#   if k.startswith('module.'):
#       # remove prefix
#       state_dict[k[len("module."):]] = state_dict[k]
#   del state_dict[k]
# for k in list(state_dict.keys()):
#     if k.startswith('module.encoder_q'):
#         # remove prefix
#         state_dict[k.replace('module.encoder_q', 'encoder')] = state_dict[k]
#     # delete renamed or unused k
#     del state_dict[k]
# for k in list(state_dict.keys()):
#     if 'fc.0' in k:
#         state_dict[k.replace('fc.0','fc1')] = state_dict[k]
#     if 'fc.2' in k:
#         state_dict[k.replace('fc.2','fc2')] = state_dict[k]
#         del state_dict[k]


# print(state_dict.keys())
#
# msg= net.load_state_dict(state_dict)
print(net)
# exit()
# print(msg.missing_keys)
# print(state_dict.keys())
print("model_loaded")

# torch.save(net.state_dict(), 'resnet50_moco_supervised_96_best.pth')

# net.load_state_dict()
net = net.cuda()

net.eval()
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

print(f"Team {team_id}: {team_name} Accuracy: {(100 * correct / total):.2f}%")
print(f"Team {team_id}: {team_name} Email: {email_address}")
