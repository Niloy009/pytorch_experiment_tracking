"""
model_builder.py

This module provides factory functions for creating and customizing EfficientNet-based
image classification models using PyTorch and torchvision. Models are initialized with 
pretrained ImageNet weights, feature extractor layers are frozen, and classification heads 
are modified for the target number of output classes.

Key Features:
-------------
- EfficientNet-B0 and EfficientNet-B2 architectures with pretrained weights.
- Freezes convolutional feature extractor layers for transfer learning.
- Customizable classifier heads for specific output classes.
- Reproducibility ensured by setting random seeds.
- Device-aware model creation (CPU/GPU).

Functions:
----------
- create_effnetb0():
    Creates a modified EfficientNet-B0 model with a frozen feature extractor
    and a custom classifier for the projects output classes.

- create_effnetb2():
    Creates a modified EfficientNet-B2 model with a frozen feature extractor
    and a custom classifier for the projects output classes.

Usage Example:
--------------
    from model_builder import create_effnetb0, create_effnetb2

    model_b0 = create_effnetb0()
    model_b2 = create_effnetb2()

Author: Niloy Saha Roy
Created: 2025-08-14
"""

import torchvision
import torch
from torch import nn
from . import utils

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get num of the output features
OUT_FEATURE = 3  # For pizza, steak, sushi


def create_effnetb0():
    """Create an EffnetB0 feature extractor"""
    # get the pretrained weights nad base model
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(DEVICE)

    # Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # set the seeds
    utils.set_seeds()

    # Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=OUT_FEATURE)
    ).to(DEVICE)

    # give the model name
    model.name = 'effnetb0'
    print(f"[INFO] Created new {model.name} model")
    return model

def create_effnetb2():
    """Create an EffnetB0 feature extractor"""

    # get the pretrained weights nad base model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(DEVICE)

    # Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # set the seeds
    utils.set_seeds()

    # Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=OUT_FEATURE)
    ).to(DEVICE)

    # give the model name
    model.name = 'effnetb2'
    print(f"[INFO] Created new {model.name} model")
    return model
