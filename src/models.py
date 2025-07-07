from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
# from transformers.models import ViTForImageClassification

class ResNet18Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet18Model, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
class CustomResNet18Model(nn.Module):
    def __init__(self, num_classes: int = 1000, *, hidden_size: int = 512, dropout: float = 0.3):
        super(CustomResNet18Model, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)  # final output layer
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class ResNet34Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet34Model, self).__init__()
        self.resnet = resnet34(weights=ResNet34Model.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
class ViT_B_16Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ViT_B_16Model, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
# class ViTModel(nn.Module):
#     def __init__(self, num_classes: int = 1000):
#         super(ViTModel, self).__init__()
#         self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
#         self.vit.classifier = nn.Linear(self.vit.classifier.in_features, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.vit(x).logits