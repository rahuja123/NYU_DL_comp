import torch
import torchvision
import torchvision.models as models
from torch import nn
from resnet_large1 import ResNet, BasicBlock, Bottleneck
from data_helper import CustomDataset
from torchvision import transforms

# model= models.resnet34()
# # model.fc= torch.nn.Identity()
# resnet34= torch.nn.Sequential(*[model.[i] for i in range(4)])
# # resnet34
# print(resnet34)


path= '/home/rahulahuja/nyu/dl/NYU_DL_comp/self-supervised-relational-reasoning/checkpoint/relationnet/nyu/resnet50/relationnet_nyu_resnet50_seed_2_epoch_19.tar'
checkpoint= torch.load(path)
print(checkpoint.keys())

feature_extractor = ResNet(Bottleneck, layers=[3, 4, 6, 3],zero_init_residual=False,
             groups=1, width_per_group=64, replace_stride_with_dilation=None,
             norm_layer=None)

print(feature_extractor)
feature_extractor.load_state_dict(checkpoint["backbone"])
model= models.resnet50(num_classes=800)
model.load_state_dict(checkpoint["backbone"],strict=False)
model.fc.load_state_dict(checkpoint["classifier"])
torch.save(model.state_dict(), 'resnet50_relation_epoch19.pth')
print(model)

# exit()

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
eval_transform = transforms.Compose([
    transforms.ToTensor(), normalize
])

evalset = CustomDataset(root='../dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

net= model
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


print((100 * correct / total))
# print(feature_extractor.fc)
