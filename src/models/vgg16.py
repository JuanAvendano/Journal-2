"""
Created on Thursday Mar 19 2026


src/models/vgg16.py
------------------------------------------------------------------------------
VGG16 architecture loader for transfer learning.

This file contains only the model definition — no training loop, no data
loading, no evaluation. All of that lives in base_model.py and is shared
across all three architectures.

About VGG16:
  VGG16 was introduced by Simonyan & Zisserman (2014) and was one of the
  first very deep networks (16 weight layers). Its architecture is a simple
  stack of 3x3 convolution layers followed by max-pooling, making it easy
  to understand and modify. The name "16" refers to the total number of
  layers with learnable weights.

  Architecture summary:
    - features: 13 convolutional layers organised in 5 blocks
    - avgpool:  adaptive average pooling
    - classifier: 3 fully connected layers
      - classifier[0]: FC 25088 → 4096
      - classifier[3]: FC 4096  → 4096
      - classifier[6]: FC 4096  → num_classes  ← we replace this

Transfer learning strategy:
  We load ImageNet pretrained weights, freeze the convolutional feature
  extractor (features), and only train the final classifier layer.
  This works well when:
    - Your dataset is relatively small (few thousand images).
    - Your images are somewhat similar to natural images (concrete textures
      share low-level features like edges and gradients with ImageNet images).
"""

import torch.nn as nn
from torchvision import models


def load_vgg16(num_classes: int, device) -> nn.Module:
    """
    Load VGG16 with ImageNet pretrained weights and replace the final
    classification layer to match the number of damage classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes, e.g. 4 for (crack, efflorescence,
        spalling, undamaged).
    device : torch.device
        The device to move the model to (CPU or CUDA GPU).

    Returns
    -------
    nn.Module
        VGG16 model ready for training.
    """
    # Load VGG16 with the latest recommended pretrained weights.
    # VGG16_Weights.DEFAULT always uses the best available weights,
    # replacing the deprecated pretrained=True argument.
    weights = models.VGG16_Weights.DEFAULT
    model   = models.vgg16(weights=weights)

    # -------------------------------------------------------------------------
    # Freeze the feature extractor (convolutional layers)
    # -------------------------------------------------------------------------
    # model.features contains all 13 convolutional layers.
    # Setting requires_grad=False tells PyTorch NOT to compute gradients
    # for these parameters — they will not be updated during training.
    # This drastically reduces training time and the amount of labelled data
    # needed, because we are only learning the final classification head.
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Replace the final classification layer
    # -------------------------------------------------------------------------
    # The original VGG16 classifier[6] outputs 1000 values (ImageNet classes).
    # We replace it with a new Linear layer that outputs num_classes values.
    # A new nn.Linear layer has requires_grad=True by default, so it will
    # be trained while the rest of the network stays frozen.
    #
    # model.classifier[6].in_features retrieves the input size of the original
    # layer (4096), so we don't have to hardcode it.
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    # Move the model to the target device (GPU if available).
    model = model.to(device)

    return model
