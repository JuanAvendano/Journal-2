"""
Created on Thursday Mar 19 2026

src/models/alexnet.py
------------------------------------------------------------------------------
AlexNet architecture loader for transfer learning.

About AlexNet:
  AlexNet was introduced by Krizhevsky, Sutskever & Hinton (2012) and was the
  model that triggered the deep learning revolution in computer vision by
  winning the ImageNet competition by a large margin. It was among the first
  models to use ReLU activations, dropout regularisation, and GPU training
  at scale.

  Compared to VGG16 and ResNet50, AlexNet is much smaller and simpler:
    - features: 5 convolutional layers
    - avgpool:  adaptive average pooling
    - classifier: 3 fully connected layers
      - classifier[1]: FC 9216 → 4096
      - classifier[4]: FC 4096 → 4096
      - classifier[6]: FC 4096 → num_classes  ← we replace this

  AlexNet is included in this ensemble because its architectural differences
  from VGG16 and ResNet50 mean it may learn different feature representations
  from the same images. Ensemble methods benefit most when the individual
  models make different types of errors — diversity in the ensemble is key.

Transfer learning strategy:
  Same as VGG16 — freeze the convolutional feature layers and only train
  the final classification head.

Input size:
  AlexNet expects 227×227 images (note: the original paper stated 224×224,
  but the correct implementation uses 227×227 to produce the expected
  feature map dimensions). This is why AlexNet has a different input_size
  in train_config.yaml compared to VGG16 and ResNet50.
"""

import torch.nn as nn
from torchvision import models


def load_alexnet(num_classes: int, device) -> nn.Module:
    """
    Load AlexNet with ImageNet pretrained weights and replace the final
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
        AlexNet model ready for training.
    """
    # Load AlexNet with the latest recommended pretrained weights.
    weights = models.AlexNet_Weights.DEFAULT
    model   = models.alexnet(weights=weights)

    # -------------------------------------------------------------------------
    # Freeze the feature extractor (convolutional layers)
    # -------------------------------------------------------------------------
    # AlexNet's convolutional layers are stored under model.features,
    # same as VGG16. We freeze them all to retain ImageNet representations.
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Replace the final classification layer
    # -------------------------------------------------------------------------
    # AlexNet's classifier[6] is the final layer, same index as VGG16.
    # in_features is 4096.
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    # Move the model to the target device.
    model = model.to(device)

    return model
