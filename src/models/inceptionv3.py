"""
Created on Monday Apr 27 2026

src/models/inceptionv3.py
------------------------------------------------------------------------------
InceptionV3 architecture loader for transfer learning.

About InceptionV3:
  InceptionV3 was introduced by Szegedy et al. (2015) in "Rethinking the
  Inception Architecture for Computer Vision". It belongs to the Inception
  family of networks, which use a fundamentally different design philosophy
  from VGG16, ResNet50, and AlexNet: instead of stacking one convolution
  at a time, each "inception module" applies multiple convolution sizes
  (1×1, 3×3, 5×5) and a pooling branch IN PARALLEL and then concatenates
  the results. This lets the network capture features at multiple spatial
  scales simultaneously without the designer having to choose one filter size.

  Key differences from the other models in this ensemble:
    - Inception modules: parallel multi-scale feature extraction
    - Factorised convolutions: 5×5 → two stacked 3×3; 3×3 → 1×3 + 3×1
      (fewer parameters, same receptive field, more non-linearity)
    - Auxiliary classifier: a second classification head attached mid-network
      during training to combat the vanishing gradient problem in deep nets.
      We DISABLE this (see the note below) so that the training loop in
      base_model.py does not need any changes.

  Architecture summary (simplified):
    - Conv + pooling stem
    - 3× InceptionA modules (mixed_5b, 5c, 5d)
    - 1× InceptionB module (mixed_6a) — reduces spatial size
    - 4× InceptionC modules (mixed_6b, 6c, 6d, 6e)
    - 1× InceptionD module (mixed_7a) — reduces spatial size
    - 2× InceptionE modules (mixed_7b, 7c)
    - Global average pooling
    - fc: 2048 → num_classes  ← we replace this

  Input size: 299×299 — fixed by the architecture.
    This is enforced in train.py via MODEL_INPUT_SIZES["inceptionv3"] = 299.

Why InceptionV3 adds diversity to the ensemble:
  VGG16 uses only 3×3 convolutions stacked sequentially.
  ResNet50 uses 1×1 + 3×3 + 1×1 bottleneck blocks with skip connections.
  AlexNet uses varying filter sizes (11×11, 5×5, 3×3) in sequence.
  InceptionV3 extracts features at multiple scales IN PARALLEL per layer.
  This fundamental difference in how features are computed means that
  InceptionV3 and the other three models are likely to make different
  errors on the same images — exactly the diversity that makes ensemble
  learning effective.

NOTE on the auxiliary classifier (AuxLogits):
  InceptionV3 was designed with an auxiliary head (a small branch that
  makes its own class prediction from an intermediate feature map) to help
  gradients reach the early layers during training. In training mode, the
  torchvision InceptionV3 forward() returns an InceptionOutputs named tuple:
      InceptionOutputs(logits=<main output>, aux_logits=<auxiliary output>)
  rather than a plain tensor, which would break the shared training loop in
  base_model.py (which expects a single tensor).

  We disable the auxiliary classifier by setting model.aux_logits = False
  immediately after loading. InceptionV3.forward() then returns a plain
  tensor in BOTH training and evaluation modes, so base_model.py requires
  ZERO changes. The AuxLogits submodule still exists in the model graph
  (its weights are loaded from the pretrained checkpoint) but is never
  called because the forward() branch is guarded by `if self.aux_logits`.

Transfer learning strategy:
  We freeze ALL parameters first, then replace the final fc layer with
  our custom head. The new head has requires_grad=True automatically,
  so only it is trained. This mirrors the ResNet50 strategy and is
  appropriate because the dataset is relatively small.
"""

import torch.nn as nn
from torchvision import models


def load_inceptionv3(num_classes: int, device) -> nn.Module:
    """
    Load InceptionV3 with ImageNet pretrained weights, disable the auxiliary
    classifier, and replace the final fc layer to match the number of damage
    classes.

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
        InceptionV3 model ready for training. The model always returns a
        plain tensor (not an InceptionOutputs named tuple) because the
        auxiliary classifier is disabled.
    """

    # -------------------------------------------------------------------------
    # Load pretrained InceptionV3
    # -------------------------------------------------------------------------
    # Inception_V3_Weights.DEFAULT gives the best available ImageNet weights.
    weights = models.Inception_V3_Weights.DEFAULT
    model   = models.inception_v3(weights=weights)

    # -------------------------------------------------------------------------
    # Disable the auxiliary classifier
    # -------------------------------------------------------------------------
    # InceptionV3.forward() checks self.aux_logits to decide whether to:
    #   a) compute the auxiliary branch and return InceptionOutputs (tuple), OR
    #   b) skip the auxiliary branch and return a plain tensor.
    # Setting it to False means the model always returns a plain tensor,
    # making it fully compatible with base_model.py without any changes there.
    model.aux_logits = False

    # -------------------------------------------------------------------------
    # Freeze all parameters
    # -------------------------------------------------------------------------
    # We freeze EVERYTHING first, then replace and unfreeze only the head.
    # This is important for InceptionV3 because its BatchNorm layers behave
    # unexpectedly when partially frozen (same reason as ResNet50).
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Replace the final classification layer
    # -------------------------------------------------------------------------
    # In InceptionV3, the final layer is called 'fc' (same naming as ResNet50).
    # model.fc.in_features is 2048 for InceptionV3.
    # Creating a new nn.Linear automatically sets requires_grad=True, so this
    # new layer — and only this layer — will be trained.
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the target device (GPU if available).
    model = model.to(device)

    return model