"""
Created on Friday May 22 2026

src/models/alexnet_v2.py
------------------------------------------------------------------------------
AlexNet architecture loader — VERSION 2 (partial fine-tuning).

Sibling of alexnet.py. The only change is the transfer-learning strategy: in
addition to the new classification head, this version unfreezes the last two
convolutional layers so they can fine-tune on the concrete-damage data.

Why (error decorrelation):
  With a fully frozen backbone all ensemble members make highly correlated
  errors, which caps how much fusion can help. AlexNet is the lightest model
  in your ensemble and (per your augmentation plan) is fed colour/texture-heavy
  transforms; letting its top conv layers adapt lets it specialise on those
  texture cues, pushing its errors away from the other architectures'.

What gets unfrozen here:
  AlexNet's feature extractor is an nn.Sequential of 13 entries (indices 0–12)
  with 5 conv layers at indices 0, 3, 6, 8, 10:

    index 0  : Conv 3   → 64   (11×11)
    index 3  : Conv 64  → 192  (5×5)
    index 6  : Conv 192 → 384  (3×3)
    index 8  : Conv 384 → 256  (3×3)   ← unfreeze (conv4)
    index 10 : Conv 256 → 256  (3×3)   ← unfreeze (conv5)

  We freeze everything up to index 7 and unfreeze index 8 onward (conv4 +
  conv5) plus the classification head. The ReLU/MaxPool layers in that slice
  carry no parameters, so toggling them is a no-op.

  To unfreeze conv3 as well, change UNFREEZE_FROM from 8 to 6.
"""

import torch.nn as nn
from torchvision import models


def load_alexnet_v2(num_classes: int, device) -> nn.Module:
    """
    Load AlexNet with ImageNet pretrained weights, replace the classification
    head, and unfreeze the last two convolutional layers for fine-tuning.

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
        AlexNet model ready for partial fine-tuning.
    """
    weights = models.AlexNet_Weights.DEFAULT
    model   = models.alexnet(weights=weights)

    # -------------------------------------------------------------------------
    # Step 1 — Freeze the entire feature extractor first.
    # -------------------------------------------------------------------------
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2 — Replace the classification head (unchanged from alexnet.py).
    # -------------------------------------------------------------------------
    in_features = model.classifier[1].in_features   # 9216
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )

    # -------------------------------------------------------------------------
    # Step 3 — Unfreeze the last two conv layers (index 8 onward).
    # -------------------------------------------------------------------------
    UNFREEZE_FROM = 8   # ← set to 6 to also unfreeze conv3
    for layer in model.features[UNFREEZE_FROM:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)
    return model
