import torch.nn as nn
from torchvision import models

def get_resnet18():
    resnet = models.resnet18(pretrained=False)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    return resnet
