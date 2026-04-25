"""
model.py
Model definitions for scene recognition.
Contains ConvNeXt Small and ResNet152 architectures used in the ensemble.
"""

import torch.nn as nn
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import resnet152, ResNet152_Weights

NUM_CLASSES = 15


def get_convnext_small(pretrained=True):
    """
    Returns ConvNeXt Small model with a custom 15-class classifier head.

    Args:
        pretrained (bool): If True, loads ImageNet pretrained weights.

    Returns:
        model (nn.Module): ConvNeXt Small model with 15-class head.
    """
    if pretrained:
        model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    else:
        model = convnext_small(weights=None)

    # Replace classifier head for 15 classes
    # ConvNeXt Small feature dim = 768
    model.classifier = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.LayerNorm(768),
        nn.Linear(768, NUM_CLASSES)
    )
    return model


def get_resnet152(pretrained=True):
    """
    Returns ResNet152 model with a custom 15-class fully connected head.

    Args:
        pretrained (bool): If True, loads ImageNet pretrained weights.

    Returns:
        model (nn.Module): ResNet152 model with 15-class head.
    """
    if pretrained:
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    else:
        model = resnet152(weights=None)

    # Replace final FC layer for 15 classes
    # ResNet152 feature dim = 2048
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model
