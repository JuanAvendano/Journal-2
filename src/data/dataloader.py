"""
Created on Thursday Mar 19 2026

src/data/dataloader.py
------------------------------------------------------------------------------
Dataset class and DataLoader setup for training, validation, and testing.

This module is a single definition of how data is loaded in the models,
and each training script simply calls get_dataloaders() from here.

Key concepts:
  - ImageFolder: a PyTorch dataset class that reads images from a folder
    structure where each subfolder is a class. It automatically assigns
    numeric labels based on alphabetical order of subfolder names.
  - DataLoader: wraps a Dataset and handles batching, shuffling, and
    parallel data loading (num_workers).
  - ImageFolderWithPaths: custom extension of ImageFolder that also
    returns the file path of each image, which we need to save to CSVs.
  - WeightedRandomSampler: oversamples underrepresented classes so that
    each batch sees roughly equal numbers of images from each class.
    Controlled by balancing.balanced_sampling in train_config.yaml.
"""

import os
import torch
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from src.data.augmentations import get_train_transforms, get_eval_transforms


# ==============================================================================
# Custom Dataset class
# ==============================================================================

class ImageFolderWithPaths(ImageFolder):
    """
    Extension of torchvision's ImageFolder that returns the image file path
    alongside the image tensor and its label.

    Standard ImageFolder returns: (image_tensor, label)
    This class returns:           (image_tensor, label, path)

    We need the path because we save predictions to CSV files indexed by
    image filename. Without the path we would not know which image each
    prediction row belongs to.

    This class inherits everything from ImageFolder and only overrides the
    __getitem__ method (the method that returns one sample at a time).
    """

    def __getitem__(self, index: int):
        """
        Return one sample from the dataset.

        Parameters
        ----------
        index : int
            The integer index of the sample to retrieve.

        Returns
        -------
        tuple : (image_tensor, label, path)
            image_tensor : torch.Tensor of shape (C, H, W)
            label        : int class index
            path         : str full file path to the image
        """
        # Call the parent class's __getitem__ to get the (image, label) tuple.
        # 'super()' refers to ImageFolder, the parent class.
        original_tuple = super().__getitem__(index)

        # self.samples is a list of (path, label) tuples maintained by
        # ImageFolder. We retrieve the path for this index here.
        path, _ = self.samples[index]

        # Concatenate the original (image, label) tuple with (path,) to produce
        # the three-element tuple (image, label, path).
        # Note: (path,) is a single-element tuple — the trailing comma is
        # required to make it a tuple rather than just parentheses.
        return original_tuple + (path,)


# ==============================================================================
# Class count reporting
# ==============================================================================

def get_class_counts(dataset: ImageFolderWithPaths) -> dict:
    """
    Count the number of images per class in a dataset.

    This is used to report dataset balance and to compute sampling weights
    for WeightedRandomSampler.

    Parameters
    ----------
    dataset : ImageFolderWithPaths
        A loaded ImageFolder-based dataset.

    Returns
    -------
    dict
        Keys are class names, values are integer counts.
        Example: {"crack": 400, "efflorescence": 300, "spalling": 200,
                  "undamaged": 100}
    """
    # dataset.targets is a list of integer labels for every image in the
    # dataset, in the same order as dataset.samples.
    # Counter counts how many times each label appears.
    label_counts = Counter(dataset.targets)

    # Map integer labels back to class names using dataset.classes.
    # dataset.classes is a list where index = label, value = class name.
    return {
        dataset.classes[label]: count
        for label, count in sorted(label_counts.items())
    }


def print_class_distribution(counts: dict, split_name: str, total: int) -> None:
    """
    Print a formatted table showing the number and percentage of images
    per class for a given dataset split, with a simple bar chart.

    Example output:
        Train set class distribution (1000 total):
          crack          :  400 images  ( 40.0%)  ██████████████████████
          efflorescence  :  300 images  ( 30.0%)  ████████████████
          spalling       :  200 images  ( 20.0%)  ███████████
          undamaged      :  100 images  ( 10.0%)  █████

    Parameters
    ----------
    counts : dict
        Class name → count, as returned by get_class_counts().
    split_name : str
        e.g. "Train", "Validation", "Test" — used in the header line.
    total : int
        Total number of images in this split.
    """
    print(f"\n  {split_name} set class distribution ({total} total):")

    # Find the longest class name to align columns.
    max_name_len = max(len(name) for name in counts)

    # Scale bar length to the majority class.
    max_count = max(counts.values())
    bar_max_len = 25

    for class_name, count in counts.items():
        pct = 100.0 * count / total
        bar_len = int(bar_max_len * count / max_count)
        bar = "█" * bar_len

        print(f"    {class_name:<{max_name_len}} : "
              f"{count:>5} images  ({pct:>5.1f}%)  {bar}")


def compute_balance_ratio(counts: dict) -> float:
    """
    Compute how imbalanced the dataset is as a ratio of max to min class count.

    A ratio of 1.0 means perfectly balanced.
    A ratio of 4.0 means the largest class has 4x more images than the smallest.

    Parameters
    ----------
    counts : dict
        Class name → count.

    Returns
    -------
    float
        max_count / min_count.
    """
    values = list(counts.values())
    return max(values) / min(values) if min(values) > 0 else float("inf")


# ==============================================================================
# Balanced sampler
# ==============================================================================

def build_weighted_sampler(dataset: ImageFolderWithPaths) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples underrepresented classes.

    How it works:
      Each image is assigned a sampling weight inversely proportional to
      how common its class is. Images from rare classes get higher weights
      and are drawn more frequently during training.

      Weight formula for class i:
          weight_i = total_images / (num_classes * count_i)

      Example: 4 classes, 1000 images, counts [400, 300, 200, 100]:
          weight_crack         = 1000 / (4 * 400) = 0.625
          weight_efflorescence = 1000 / (4 * 300) = 0.833
          weight_spalling      = 1000 / (4 * 200) = 1.250
          weight_undamaged     = 1000 / (4 * 100) = 2.500

      The sampler draws 'total_images' samples per epoch with replacement,
      so the number of training steps per epoch stays the same as without
      balancing — only the class distribution within those steps changes.

    Important: this does NOT create new images. It changes how frequently
    existing images are drawn during each epoch.

    Parameters
    ----------
    dataset : ImageFolderWithPaths
        The training dataset.

    Returns
    -------
    WeightedRandomSampler
        Ready to pass to the DataLoader's 'sampler' argument.
    """
    label_counts = Counter(dataset.targets)
    num_classes = len(dataset.classes)
    total = len(dataset)

    # Compute per-class weights — rare classes get higher weights.
    class_weights = {
        label: total / (num_classes * count)
        for label, count in label_counts.items()
    }

    # Assign each individual image its class weight.
    # dataset.targets[i] is the class label of the i-th image.
    sample_weights = [class_weights[label] for label in dataset.targets]

    # WeightedRandomSampler requires a DoubleTensor (float64).
    weights_tensor = torch.DoubleTensor(sample_weights)

    # num_samples=total keeps the epoch length the same as without balancing.
    # replacement=True allows the same image to be drawn multiple times per
    # epoch, which is necessary for oversampling minority classes.
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=total,
        replacement=True
    )

# ==============================================================================
# DataLoader factory
# ==============================================================================

def get_dataloaders(config: dict) -> dict:
    """
    Build and return train, validation, and test DataLoaders from a config dict.

    This is the main function called by scripts/train.py. It reads all
    necessary settings from the config and returns a dictionary of DataLoaders
    ready to use in the training loop.

    Parameters
    ----------
    config : dict
        The training config loaded from configs/train_config.yaml.
        Must contain 'paths', 'model', 'training', and 'dataset' sections.

    Returns
    -------
    dict with keys "train", "val", "test"
        Each value is a torch.utils.data.DataLoader.
    """
    # -------------------------------------------------------------------------
    # Extract settings from config
    # -------------------------------------------------------------------------
    paths      = config["paths"]
    model_cfg  = config["model"]
    train_cfg  = config["training"]

    train_path = paths["train"]
    val_path   = paths["val"]
    test_path  = paths["test"]
    batch_size = train_cfg["batch_size"]
    input_size = model_cfg["input_size"]

    balanced_sampling = config.get("balancing_sampling", {}).get("enabled", False)
    # -------------------------------------------------------------------------
    # Validate that data directories exist
    # -------------------------------------------------------------------------
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"The {name} dataset path does not exist: '{path}'. "
                f"Please update the paths section in train_config.yaml."
            )

    # -------------------------------------------------------------------------
    # Build transforms
    # -------------------------------------------------------------------------
    # Training images get the full augmentation pipeline.
    # Validation and test images get only resize + normalise.
    train_transform = get_train_transforms(input_size, config)
    eval_transform  = get_eval_transforms(input_size, config)

    # -------------------------------------------------------------------------
    # Load datasets using ImageFolderWithPaths
    # -------------------------------------------------------------------------
    # ImageFolder reads the subfolder structure automatically:
    #   data/train/crack/       → label 0
    #   data/train/efflorescence/ → label 1
    #   data/train/spalling/    → label 2
    #   data/train/undamaged/   → label 3
    # The numeric label assignment is alphabetical, so the order depends on
    # folder names. That is why class_names in the config must match the
    # alphabetical order of the subfolders.
    train_dataset = ImageFolderWithPaths(root=train_path, transform=train_transform)
    val_dataset   = ImageFolderWithPaths(root=val_path,   transform=eval_transform)
    test_dataset  = ImageFolderWithPaths(root=test_path,  transform=eval_transform)

    # -------------------------------------------------------------------------
    # Validate that dataset class order matches the config
    # -------------------------------------------------------------------------
    # train_dataset.classes is the list of class names in the order ImageFolder
    # assigned labels. We compare this to the config to catch mismatches early.
    config_classes  = config["dataset"]["class_names"]
    dataset_classes = train_dataset.classes  # e.g. ['crack', 'efflorescence', ...]

    if dataset_classes != config_classes:
        raise ValueError(
            f"Class order mismatch!\n"
            f"  Dataset folders (alphabetical): {dataset_classes}\n"
            f"  Config class_names:             {config_classes}\n"
            f"Please update 'class_names' in train_config.yaml to match the "
            f"alphabetical order of your dataset subfolders."
        )

    # -------------------------------------------------------------------------
    # Count and report class distribution for all three splits
    # -------------------------------------------------------------------------
    train_counts = get_class_counts(train_dataset)
    val_counts = get_class_counts(val_dataset)
    test_counts = get_class_counts(test_dataset)

    print(f"\n{'=' * 55}")
    print(f"  DATASET SUMMARY")
    print(f"{'=' * 55}")

    print_class_distribution(train_counts, "Train", len(train_dataset))
    print_class_distribution(val_counts, "Validation", len(val_dataset))
    print_class_distribution(test_counts, "Test", len(test_dataset))

    # Report the imbalance ratio for the training set and advise accordingly.
    ratio = compute_balance_ratio(train_counts)
    print(f"\n  Train imbalance ratio (max/min): {ratio:.2f}x")

    if ratio > 2.0 and not balanced_sampling:
        print(f"  WARNING: Imbalance ratio > 2.0. Consider setting "
              f"balancing.balanced_sampling: true in train_config.yaml.")
    elif balanced_sampling:
        print(f"  Balanced sampling: ENABLED")
    else:
        print(f"  Balanced sampling: disabled (dataset is reasonably balanced)")

    print(f"{'=' * 55}\n")

    # -------------------------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------------------------
    # When using WeightedRandomSampler, shuffle must be False — the sampler
    # itself handles randomisation. Passing shuffle=True alongside a sampler
    # raises an error in PyTorch.
    # Val and test loaders are never balanced — evaluation must reflect the
    # real class distribution to give meaningful metrics.
    # num_workers controls how many CPU subprocesses are used to load data
    # in parallel while the GPU trains. On Windows, num_workers > 0 can cause
    # issues with some setups — if you get errors, set it to 0.
    # shuffle=True for training ensures images are in a random order each epoch.
    # shuffle=False for val/test ensures deterministic evaluation.

    if balanced_sampling:
        sampler = build_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # must be False when using a sampler
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,         # Set to 0 for Windows compatibility
            pin_memory=True        # pin_memory=True speeds up CPU-to-GPU transfers
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
    }


def get_deploy_dataloader(image_dir: str, config: dict) -> DataLoader:
    """
    Build a DataLoader for deployment (inference on unlabelled images).

    In deployment mode there are no class subfolders — all images live in a
    single directory. We still use ImageFolderWithPaths so we can track which
    image each prediction belongs to, but we wrap the image directory in a
    single dummy class folder to satisfy ImageFolder's expected structure.

    Note: this function expects that the input directory has already been
    wrapped in a single subfolder (e.g. input_images/images/img001.jpg).
    scripts/deploy.py handles this wrapping automatically.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing images for inference.
    config : dict
        Deploy config loaded from configs/deploy_config.yaml.

    Returns
    -------
    DataLoader
    """
    input_size = config["input_sizes"]["vgg16"]   # Deployment uses VGG16 size as default

    norm_cfg = config.get("normalisation", {})

    transform = get_eval_transforms(input_size, {"normalisation": norm_cfg})

    dataset = ImageFolderWithPaths(root=image_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return loader
