from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

class ResNet18Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet18Model, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class ResNet34Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet34Model, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)