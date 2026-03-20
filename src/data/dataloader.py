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
"""

import os
from pathlib import Path
from torch.utils.data import DataLoader
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
    # Print dataset summary
    # -------------------------------------------------------------------------
    print(f"Classes detected: {dataset_classes}")
    print(f"  Train samples:      {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples:       {len(test_dataset)}")

    # -------------------------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------------------------
    # num_workers controls how many CPU subprocesses are used to load data
    # in parallel while the GPU trains. On Windows, num_workers > 0 can cause
    # issues with some setups — if you get errors, set it to 0.
    # shuffle=True for training ensures images are in a random order each epoch.
    # shuffle=False for val/test ensures deterministic evaluation.

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

    # Use eval transforms (no augmentation) — we are doing inference, not training.
    norm_cfg = config.get("normalisation", {})
    from src.data.augmentations import get_eval_transforms
    eval_transform = get_eval_transforms(input_size, {"normalisation": norm_cfg})

    dataset = ImageFolderWithPaths(root=image_dir, transform=eval_transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return loader
