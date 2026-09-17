"""
Created on Friday May 22 2026

src/models/resnet50_v2.py
------------------------------------------------------------------------------
ResNet50 architecture loader — VERSION 2 (partial fine-tuning).

Sibling of resnet50.py. The only difference is the transfer-learning strategy:
in addition to training the new fc head, this version unfreezes layer4 — the
final residual stage — so it can adapt to the concrete-damage data.

Why (error decorrelation):
  A frozen ImageNet backbone makes every ensemble member extract near-identical
  features, producing correlated errors that fusion methods cannot fix. Letting
  ResNet50 fine-tune its highest-level residual stage gives it room to learn
  features specific to your damage classes that differ from what VGG16,
  AlexNet, EfficientNet and InceptionV3 learn — increasing the diversity that
  soft voting, Bayesian fusion, Sugeno and the MLP meta-learner depend on.

What gets unfrozen here:
  ResNet50 layout: conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 →
  layer4 → avgpool → fc. We freeze everything, replace fc, then unfreeze the
  whole of layer4 (the last group of bottleneck residual blocks) + fc.

  To go deeper, also unfreeze layer3 (add `model.layer3` to the loop below).

A note on BatchNorm:
  layer4 contains BatchNorm layers. Because we call model.train() during the
  training loop, these BN layers will both update their learnable affine
  parameters (gamma/beta, now trainable) AND recompute their running
  mean/variance from your batches — which is what we want for adaptation.
  The frozen earlier layers keep their ImageNet running statistics. This is
  consistent with the original (frozen) ResNet50 setup.
"""

import torch.nn as nn
from torchvision import models


def load_resnet50_v2(num_classes: int, device) -> nn.Module:
    """
    Load ResNet50 with ImageNet pretrained weights, replace the fc head, and
    unfreeze the final residual stage (layer4) for fine-tuning.

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
        ResNet50 model ready for partial fine-tuning.
    """
    weights = models.ResNet50_Weights.DEFAULT
    model   = models.resnet50(weights=weights)

    # -------------------------------------------------------------------------
    # Step 1 — Freeze ALL parameters.
    # -------------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Step 2 — Replace the final fully connected layer (unchanged).
    # -------------------------------------------------------------------------
    # The new nn.Linear is trainable by default (requires_grad=True).
    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Linear(in_features, num_classes)

    # -------------------------------------------------------------------------
    # Step 3 — Unfreeze the last residual stage (layer4).
    # -------------------------------------------------------------------------
    # layer4 is the deepest group of bottleneck blocks and produces the most
    # task-specific, high-level features — the best candidate for fine-tuning.
    for param in model.layer4.parameters():     # add model.layer3 here to go deeper
        param.requires_grad = True

    model = model.to(device)
    return model
