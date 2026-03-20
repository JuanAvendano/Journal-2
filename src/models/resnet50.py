"""
Created on Thursday Mar 19 2026

src/models/resnet50.py
------------------------------------------------------------------------------
ResNet50 architecture loader for transfer learning.

About ResNet50:
  ResNet50 was introduced by He et al. (2015) and solved the "vanishing
  gradient" problem that prevented training very deep networks. It introduced
  residual connections (also called skip connections): the input of a block
  is added directly to the output, creating a shortcut path for gradients
  to flow through during backpropagation.

  The "50" refers to 50 weight layers. The architecture uses "bottleneck"
  blocks (1x1 → 3x3 → 1x1 convolutions) to keep the number of parameters
  manageable despite the depth.

  Architecture summary:
    - conv1:   initial 7x7 convolution
    - layer1–4: four groups of residual bottleneck blocks
    - avgpool:  global average pooling (reduces spatial dims to 1x1)
    - fc:       single fully connected layer, 2048 → num_classes ← we replace this

Transfer learning strategy:
  ResNet50 uses a slightly different freezing strategy than VGG16.
  We freeze ALL parameters first, then unfreeze only the final fc layer.
  This is because ResNet's batch normalisation layers behave unexpectedly
  when partially frozen — freezing everything and then selectively unfreezing
  is cleaner.
"""

import torch.nn as nn
from torchvision import models


def load_resnet50(num_classes: int, device) -> nn.Module:
    """
    Load ResNet50 with ImageNet pretrained weights and replace the final
    fully connected layer to match the number of damage classes.

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
        ResNet50 model ready for training.
    """
    # Load ResNet50 with the latest recommended pretrained weights.
    weights = models.ResNet50_Weights.DEFAULT
    model   = models.resnet50(weights=weights)

    # -------------------------------------------------------------------------
    # Freeze all parameters first
    # -------------------------------------------------------------------------
    # model.parameters() returns all learnable parameters in the network.
    # We freeze everything in one pass, then selectively unfreeze below.
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Replace the final fully connected layer
    # -------------------------------------------------------------------------
    # In ResNet50, the final layer is called 'fc' (not 'classifier' like VGG16).
    # model.fc.in_features is 2048 for ResNet50.
    in_features = model.fc.in_features
    # Creating a new nn.Linear automatically sets requires_grad=True,
    # so this layer will be trained while everything else remains frozen.
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the target device.
    model = model.to(device)

    return model
