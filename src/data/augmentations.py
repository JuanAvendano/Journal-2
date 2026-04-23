"""
src/data/augmentations.py
------------------------------------------------------------------------------
Data augmentation transforms for training.

Data augmentation artificially increases the diversity of the training set by
applying random transformations to images. The key rule is:
  - Augmentation is applied ONLY to the training set.
  - Validation and test sets always use only the basic transforms (resize +
    normalise), so that evaluation results are consistent and comparable.

Why augment?
  - Concrete damage datasets tend to be relatively small.
  - Augmentation helps the model generalise better to unseen images taken from
    different angles, lighting conditions, or camera distances.
  - It reduces overfitting, where the model memorises training images instead
    of learning general features.

This module provides two functions:
  - get_train_transforms(): full augmentation pipeline for training
  - get_eval_transforms():  minimal pipeline for validation and test sets
"""

from torchvision import transforms


def get_train_transforms(input_size: int, config: dict) -> transforms.Compose:
    """
    Build the augmentation + preprocessing pipeline for the training set.

    The pipeline is controlled by the 'augmentation' section of train_config.yaml.
    If augmentation is disabled in the config, only the basic resize and
    normalise steps are applied (same as the eval pipeline).

    The order of transforms matters:
      1. Resize first (so all images are the same size before other transforms).
      2. Geometric transforms (flip, rotation) — these change spatial structure.
      3. Colour transforms — these change pixel values.
      4. ToTensor() — converts PIL Image (H, W, C) to PyTorch tensor (C, H, W)
         and scales pixel values from [0, 255] to [0.0, 1.0].
      5. Normalize() — shifts the distribution to match ImageNet statistics.

    Parameters
    ----------
    input_size : int
        The spatial size to resize images to (e.g. 224 for VGG16/ResNet50,
        227 for AlexNet). Height and width are set to the same value.
    config : dict
        The full training config loaded from train_config.yaml. The function
        reads the 'augmentation' and 'normalisation' sections.

    Returns
    -------
    transforms.Compose
        A composed transform pipeline ready to pass to ImageFolder.
    """
    aug_cfg  = config.get("augmentation", {})
    norm_cfg = config.get("normalisation", {})

    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std  = norm_cfg.get("std",  [0.229, 0.224, 0.225])

    # Start with the mandatory resize step.
    # transforms.Resize((h, w)) always resizes to exactly (h, w).
    transform_list = [transforms.Resize((input_size, input_size))]

    # -------------------------------------------------------------------------
    # Geometric augmentations (only added if enabled in config)
    # -------------------------------------------------------------------------

    if aug_cfg.get("enabled", True):

        if aug_cfg.get("horizontal_flip", True):
            # Randomly flips the image left-to-right with 50% probability.
            # Useful for damage images because damage can appear on either side.
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if aug_cfg.get("vertical_flip", False):
            # Vertical flipping is off by default — an upside-down bridge image
            # is not physically meaningful, but you can enable it if needed.
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        rotation = aug_cfg.get("rotation_degrees", 0)
        if rotation > 0:
            # Randomly rotates the image by up to ±rotation_degrees.
            # fill=0 means the corners introduced by rotation are filled with
            # black pixels. This simulates images taken at slight angles.
            transform_list.append(
                transforms.RandomRotation(degrees=rotation, fill=0)
            )
        if aug_cfg.get("random_perspective", False):
            distortion = aug_cfg.get("perspective_distortion", 0.3)
            # Applies a random perspective (projective) transformation.
            # This simulates off-angle camera shots — common in field inspection
            # where the camera is rarely perfectly parallel to the surface.
            # distortion_scale controls how extreme the warping can be:
            #   0.0 = no change, 1.0 = very extreme warp. 0.2-0.4 is realistic.
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=distortion,
                    p=0.5  # Applied with 50% probability per image
                )
            )
    # -------------------------------------------------------------------------
    # Colour augmentations
    # -------------------------------------------------------------------------

        if aug_cfg.get("colour_jitter", False):
            strength = aug_cfg.get("colour_jitter_strength", 0.2)
            # ColorJitter randomly varies brightness, contrast, and saturation.
            # 'strength' controls how much variation is applied.
            # This simulates different lighting conditions on site.
            transform_list.append(
                transforms.ColorJitter(
                    brightness=strength,
                    contrast=strength,
                    saturation=strength,
                    hue=0.0    # Hue shift is kept at 0 — it can change colours
                               # too dramatically for concrete damage images.
                )
            )
        if aug_cfg.get("random_grayscale", False):
            probability = aug_cfg.get("grayscale_probability", 0.2)
            # Randomly converts the image to grayscale with the given probability.
            # The output still has 3 channels (values are equal across R, G, B)
            # so it remains compatible with models pretrained on RGB ImageNet.
            # Useful for making the model robust to low-colour inspection photos.
            transform_list.append(
                transforms.RandomGrayscale(p=probability)
            )
    # -------------------------------------------------------------------------
    # Mandatory final steps (always applied, regardless of augmentation setting)
    # -------------------------------------------------------------------------

    # ToTensor() converts a PIL Image (H x W x C, uint8 [0,255]) to a
    # PyTorch FloatTensor (C x H x W, float32 [0.0, 1.0]).
    # This reordering of dimensions (HWC → CHW) is required by PyTorch.
    transform_list.append(transforms.ToTensor())

    # Normalize() applies: output = (input - mean) / std  per channel.
    # This centres the pixel distribution around zero and scales it to
    # roughly unit variance, which helps gradient-based optimisation.
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_eval_transforms(input_size: int, config: dict) -> transforms.Compose:
    """
    Build the preprocessing pipeline for validation and test sets.

    No augmentation is applied here — only the mandatory resize and normalise
    steps. This ensures that evaluation metrics are deterministic and not
    affected by random transformations.

    Parameters
    ----------
    input_size : int
        Same value as used for training (224 or 227).
    config : dict
        The full training config (only 'normalisation' section is used here).

    Returns
    -------
    transforms.Compose
    """
    norm_cfg = config.get("normalisation", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std  = norm_cfg.get("std",  [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
