import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

def get_model(num_classes=10):
    model = deeplabv3_resnet101(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model