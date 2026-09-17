"""
Created on Friday May 22 2026

src/models/vgg16_v2.py
------------------------------------------------------------------------------
VGG16 architecture loader — VERSION 2 (partial fine-tuning).

This is a sibling of vgg16.py. The ONLY difference is the transfer-learning
strategy: instead of freezing the entire convolutional feature extractor and
training only the classification head, this version ALSO unfreezes the last
convolutional block so it can fine-tune on the concrete-damage data.

Why unfreeze more than the head (the reason you asked for this):
  When every model in the ensemble shares a fully frozen ImageNet backbone,
  they all extract almost identical features, so they tend to make the SAME
  mistakes on the same images. Those "correlated errors" are exactly what
  undermines ensemble fusion (soft voting, Bayesian, Sugeno, the MLP
  meta-learner) — fusion only helps when the base learners fail in DIFFERENT
  ways. Letting each architecture fine-tune its HIGH-LEVEL layers on your data
  allows each one's inductive bias to shape distinct high-level features,
  which decorrelates their errors. We keep the low/mid-level layers frozen
  because those generic edge/texture detectors transfer well and re-training
  them on a few thousand images would risk overfitting and catastrophic
  forgetting.

What gets unfrozen here:
  VGG16's feature extractor is a single nn.Sequential of 31 entries (indices
  0–30) organised into 5 conv blocks separated by max-pool layers:

    block 1: indices 0–4    (2 conv layers)
    block 2: indices 5–9    (2 conv layers)
    block 3: indices 10–16  (3 conv layers)
    block 4: indices 17–23  (3 conv layers)
    block 5: indices 24–30  (3 conv layers + max-pool)   ← we UNFREEZE this

  So we freeze indices 0–23 (blocks 1–4) and unfreeze indices 24 onward
  (block 5, the three highest-level 512-channel conv layers) plus the
  classification head.

To unfreeze MORE (e.g. blocks 4 + 5), change the slice start from 24 to 17.
To unfreeze LESS, raise it. The optimizer in scripts/train.py automatically
trains whatever has requires_grad=True, so no other file needs editing.
"""

import torch.nn as nn
from torchvision import models


def load_vgg16_v2(num_classes: int, device) -> nn.Module:
    """
    Load VGG16 with ImageNet pretrained weights, replace the classification
    head, and unfreeze the LAST convolutional block (block 5) for fine-tuning.

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
        VGG16 model ready for partial fine-tuning.
    """
    weights = models.VGG16_Weights.DEFAULT
    model   = models.vgg16(weights=weights)

    # -------------------------------------------------------------------------
    # Step 1 — Freeze the ENTIRE feature extractor first.
    # -------------------------------------------------------------------------
    # We start from a fully frozen backbone, then selectively switch the last
    # block back on. Freezing first and unfreezing second is easier to read
    # and makes the "what is trainable" decision explicit in one place.
    for param in model.features.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2 — Replace the classification head (unchanged from vgg16.py).
    # -------------------------------------------------------------------------
    # A freshly created nn.Linear has requires_grad=True by default, so this
    # whole new head is trainable automatically.
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(p=0.65),
        nn.Linear(512, num_classes)
    )

    # -------------------------------------------------------------------------
    # Step 3 — Unfreeze the last convolutional block (block 5 = indices 24+).
    # -------------------------------------------------------------------------
    # model.features[24:] returns a Sequential SLICE containing the layers from
    # index 24 to the end. Iterating over it yields each layer; ReLU and
    # MaxPool layers have no parameters, so setting requires_grad on them is a
    # harmless no-op — only the three Conv2d layers actually become trainable.
    UNFREEZE_FROM = 24   # ← lower this number to unfreeze more conv blocks
    for layer in model.features[UNFREEZE_FROM:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)
    return model
