"""
Created on Friday May 22 2026

src/models/efficientnet_b0_v2.py
------------------------------------------------------------------------------
EfficientNet-B0 architecture loader — VERSION 2 (partial fine-tuning).

Sibling of efficientnet_b0.py. The only change is the transfer-learning
strategy: in addition to the new classifier head, this version unfreezes the
last MBConv stages (including their Squeeze-and-Excitation blocks) and the
final 1×1 conv head, so they can fine-tune on the concrete-damage data.

Why (error decorrelation):
  A frozen backbone makes ensemble members produce correlated errors that
  fusion cannot repair. EfficientNet is the only model in your ensemble whose
  blocks contain SE channel-attention; fine-tuning the top stages lets that
  attention re-weight channels for YOUR damage classes, producing features —
  and therefore errors — that differ from the other four architectures.

What gets unfrozen here:
  EfficientNet-B0's feature extractor is an nn.Sequential of 9 entries
  (indices 0–8):

    index 0 : Conv stem
    index 1 : MBConv stage 1
    ...
    index 6 : MBConv stage 6   ← unfreeze
    index 7 : MBConv stage 7   ← unfreeze
    index 8 : 1×1 conv head → 1280 channels   ← unfreeze

  We freeze indices 0–5 and unfreeze index 6 onward (the two highest-capacity
  MBConv stages + the conv head) plus the classification head. The MBConv
  stages contain BatchNorm; as with ResNet50, model.train() lets the unfrozen
  BN layers update both their affine parameters and running statistics.

  To unfreeze fewer stages, raise UNFREEZE_FROM; to unfreeze more, lower it.
"""

import torch.nn as nn
from torchvision import models


def load_efficientnet_b0_v2(num_classes: int, device) -> nn.Module:
    """
    Load EfficientNet-B0 with ImageNet pretrained weights, replace the final
    classification layer, and unfreeze the top MBConv stages + conv head for
    fine-tuning.

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
        EfficientNet-B0 model ready for partial fine-tuning.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model   = models.efficientnet_b0(weights=weights)

    # -------------------------------------------------------------------------
    # Step 1 — Freeze the entire feature extractor first.
    # -------------------------------------------------------------------------
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2 — Replace the final classification layer (unchanged).
    # -------------------------------------------------------------------------
    # We keep model.classifier[0] (Dropout) and replace model.classifier[1]
    # (the Linear layer). The new Linear is trainable by default.
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # -------------------------------------------------------------------------
    # Step 3 — Unfreeze the top stages of the backbone (index 6 onward).
    # -------------------------------------------------------------------------
    UNFREEZE_FROM = 6   # ← lower this to unfreeze more MBConv stages
    for block in model.features[UNFREEZE_FROM:]:
        for param in block.parameters():
            param.requires_grad = True

    model = model.to(device)
    return model
