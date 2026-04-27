"""
Created on Monday Apr 27 2026

src/models/efficientnet_b0.py
------------------------------------------------------------------------------
EfficientNet-B0 architecture loader for transfer learning.

About EfficientNet-B0:
  EfficientNet was introduced by Tan & Le (2019) in "EfficientNet: Rethinking
  Model Scaling for Convolutional Neural Networks". The core idea is that
  CNNs can be scaled more efficiently by jointly increasing width (channels),
  depth (layers), and resolution (input size) using a fixed ratio — called
  compound scaling — rather than scaling only one dimension at a time.

  EfficientNet-B0 is the baseline model in the family (B0 through B7).
  Despite being the smallest variant, it achieves strong accuracy while
  using far fewer parameters than VGG16 or ResNet50.

  The building block is the MBConv (Mobile Inverted Bottleneck Convolution),
  inherited from MobileNetV2. Each MBConv:
    1. Expands the channel count with a 1×1 pointwise convolution.
    2. Applies depthwise separable convolution (one filter per channel).
    3. Projects back to a smaller channel count with a 1×1 convolution.
    4. Includes a Squeeze-and-Excitation (SE) block that recalibrates
       channel-wise feature responses — essentially a learned attention
       mechanism over channels.

  Architecture summary (simplified):
    - features: 9 MBConv stages with increasing channels and strides
    - avgpool:  adaptive average pooling → (1, 1) spatial size
    - classifier:
        [0]: Dropout(p=0.2)
        [1]: Linear(1280, num_classes)  ← we replace this

  Input size: 224×224 (B0 uses the same size as VGG16 and ResNet50).

Why EfficientNet-B0 adds diversity to the ensemble:
  VGG16:         sequential 3×3 convolutions → spatial hierarchy
  ResNet50:      residual skip connections → gradient highway
  AlexNet:       varying filter sizes in sequence
  InceptionV3:   parallel multi-scale feature extraction
  EfficientNet:  depthwise separable convolutions + channel attention (SE)
                 The SE blocks make this the only model in the ensemble that
                 learns which channels are most informative per spatial region.
                 This is qualitatively different from the attention mechanisms
                 in the other architectures, adding genuine independent errors.

Transfer learning strategy:
  We freeze all parameters in model.features (the convolutional backbone)
  and replace model.classifier[1] (the final linear layer). The new layer
  has requires_grad=True by default, so only it is trained. This is
  consistent with the VGG16 freezing strategy and appropriate for a
  small dataset.

  Note: We leave model.classifier[0] (Dropout) in place. Dropout provides
  regularisation on the features from the frozen backbone before the new
  classification head, which helps prevent overfitting on the new task.
"""

import torch.nn as nn
from torchvision import models


def load_efficientnet_b0(num_classes: int, device) -> nn.Module:
    """
    Load EfficientNet-B0 with ImageNet pretrained weights, freeze the
    convolutional backbone, and replace the final classification layer to
    match the number of damage classes.

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
        EfficientNet-B0 model ready for training.
    """

    # -------------------------------------------------------------------------
    # Load pretrained EfficientNet-B0
    # -------------------------------------------------------------------------
    # EfficientNet_B0_Weights.DEFAULT selects the best available ImageNet weights.
    # EfficientNet-B0 was available in torchvision from version 0.13 onward.
    # Your CNNenv has torchvision 0.25.0, so this is fully supported.
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model   = models.efficientnet_b0(weights=weights)

    # -------------------------------------------------------------------------
    # Freeze the feature extractor (convolutional backbone)
    # -------------------------------------------------------------------------
    # model.features contains all 9 MBConv stages including the SE blocks.
    # Setting requires_grad=False prevents gradients from flowing through
    # these layers, so their weights are not updated during training.
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Replace the final classification layer
    # -------------------------------------------------------------------------
    # EfficientNet-B0's classifier is a two-element Sequential:
    #   model.classifier[0] = Dropout(p=0.2)   ← keep this for regularisation
    #   model.classifier[1] = Linear(1280, 1000) ← replace this
    #
    # model.classifier[1].in_features is 1280 for all B0 variants.
    # We retrieve it dynamically so the code still works if you swap to
    # a larger EfficientNet variant (B1, B2, ...) with a different head size.
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    # The new Linear layer has requires_grad=True by default —
    # only this layer will be trained.

    # Move the model to the target device.
    model = model.to(device)

    return model