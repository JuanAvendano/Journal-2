"""
Created on Friday May 22 2026

src/models/inceptionv3_v2.py
------------------------------------------------------------------------------
InceptionV3 architecture loader — VERSION 2 (partial fine-tuning).

Sibling of inceptionv3.py. The auxiliary classifier is still disabled (so the
model returns a plain tensor and base_model.py needs no changes). The only
difference is the transfer-learning strategy: in addition to the new fc head,
this version unfreezes the three highest-level inception modules so they can
fine-tune on the concrete-damage data.

Why (error decorrelation):
  Sharing a frozen ImageNet backbone across all five models produces correlated
  errors that fusion methods cannot overcome. InceptionV3 computes features at
  multiple scales in parallel; fine-tuning its top inception modules lets that
  multi-scale machinery specialise to your damage classes, diversifying its
  errors relative to the other architectures.

What gets unfrozen here:
  The top of InceptionV3 is: ... → Mixed_7a (InceptionD) → Mixed_7b (InceptionE)
  → Mixed_7c (InceptionE) → avgpool → dropout → fc. We freeze everything,
  replace fc, then unfreeze Mixed_7a, Mixed_7b and Mixed_7c + fc.

  To go deeper, add the Mixed_6* modules (e.g. model.Mixed_6e) to the list.

BatchNorm note:
  InceptionV3 uses BatchNorm throughout. The unfrozen Mixed_7* modules will
  update both their affine parameters and running statistics during
  model.train(), which is the desired behaviour for adaptation. This mirrors
  the ResNet50 / EfficientNet handling.
"""

import torch.nn as nn
from torchvision import models


def load_inceptionv3_v2(num_classes: int, device) -> nn.Module:
    """
    Load InceptionV3 with ImageNet pretrained weights, disable the auxiliary
    classifier, replace the fc head, and unfreeze the top inception modules
    for fine-tuning.

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
        InceptionV3 model ready for partial fine-tuning. Always returns a plain
        tensor because the auxiliary classifier is disabled.
    """
    weights = models.Inception_V3_Weights.DEFAULT
    model   = models.inception_v3(weights=weights)

    # -------------------------------------------------------------------------
    # Disable the auxiliary classifier (same as inceptionv3.py).
    # -------------------------------------------------------------------------
    # Keeps forward() returning a single tensor so base_model.py is unchanged.
    model.aux_logits = False

    # -------------------------------------------------------------------------
    # Step 1 — Freeze ALL parameters.
    # -------------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2 — Replace the final fully connected layer (unchanged).
    # -------------------------------------------------------------------------
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Linear(in_features, num_classes)

    # -------------------------------------------------------------------------
    # Step 3 — Unfreeze the three highest-level inception modules.
    # -------------------------------------------------------------------------
    # Mixed_7a/7b/7c are the deepest feature-mixing blocks before the classifier.
    top_modules = [model.Mixed_7a, model.Mixed_7b, model.Mixed_7c]
    for module in top_modules:                  # add model.Mixed_6e etc. to go deeper
        for param in module.parameters():
            param.requires_grad = True

    model = model.to(device)
    return model
