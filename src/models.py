from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet, resnet18, resnet34, resnet50
from torchvision.models import resnext50_32x4d, convnext_tiny
from torchvision.models import efficientnet_v2_s
from torchvision.models import inception_v3
from torchvision.models import vit_b_16
from torchvision.models import swin_t
from ultralytics import YOLO
# from transformers.models import ViTForImageClassification

class ResNet18Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet18Model, self).__init__()
        self.resnet = resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
class CustomResNet18Model(nn.Module):
    def __init__(self, num_classes: int = 1000, *, hidden_size: int = 512, dropout: float = 0.3):
        super(CustomResNet18Model, self).__init__()
        self.resnet = resnet18(weights='DEFAULT')
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
        self.resnet = resnet34(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNet50Model, self).__init__()
        self.resnet = resnet50(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
class ResNext50Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ResNext50Model, self).__init__()
        self.resnext = resnext50_32x4d(weights='DEFAULT')
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnext(x)
class ConvNeXt_TinyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ConvNeXt_TinyModel, self).__init__()
        self.convnext = convnext_tiny(weights='DEFAULT')
        self.convnext.classifier[-1] = nn.Linear(self.convnext.classifier[-1].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)
    
class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(EfficientNetV2Model, self).__init__()
        self.efficientnet = efficientnet_v2_s(weights='DEFAULT')
        self.efficientnet.classifier[-1] = nn.Linear(self.efficientnet.classifier[-1].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)    
class ViT_B_16_Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ViT_B_16_Model, self).__init__()
        self.vit = vit_b_16(weights='DEFAULT')
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
class Swin_T_Model(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(Swin_T_Model, self).__init__()
        self.swin = swin_t(weights='DEFAULT')
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swin(x)
    
# class ViTModel(nn.Module):
#     def __init__(self, num_classes: int = 1000):
#         super(ViTModel, self).__init__()
#         self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
#         self.vit.classifier = nn.Linear(self.vit.classifier.in_features, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.vit(x).logits

class UltralyticsYOLO(nn.Module):
    def __init__(self, num_classes: int = 1000, model_path='models/yolo11s-cls.pt', **kwargs):
        super(UltralyticsYOLO, self).__init__()
        self.model = YOLO(model_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = self.model(x)
        return results