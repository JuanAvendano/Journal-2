"""
Created on Thursday Mar 19 2026


test.py
------------------------------------------------------------------------------
Test ti verify that all the modules import without errors.

"""
import torch
from src.utils.io_utils import load_config, make_run_dir
from src.utils.logger import get_logger
from src.data.augmentations import get_train_transforms, get_eval_transforms
from src.data.dataloader import get_dataloaders, ImageFolderWithPaths
from src.models.vgg16 import load_vgg16
from src.models.resnet50 import load_resnet50
from src.models.alexnet import load_alexnet
from src.models.base_model import train_model, evaluate_final_model


# ------------------------------------------------------------------------------
# config loads nad paths
# ------------------------------------------------------------------------------

config = load_config("configs/train_config.yaml")
print(config["model"]["name"])         # should print "vgg16"
print(config["dataset"]["class_names"]) # should print the four classes
print(config["training"]["epochs"])    # should print 50

# ------------------------------------------------------------------------------
# Dataloader
# ------------------------------------------------------------------------------

config = load_config("configs/train_config.yaml")
loaders = get_dataloaders(config)

# Grab one batch from the train loader and inspect it
batch = next(iter(loaders["train"]))
images, labels, paths = batch

print(f"Batch image shape: {images.shape}")   # should be [32, 3, 224, 224]
print(f"Batch labels: {labels}")              # should be integers 0-3
print(f"First image path: {paths[0]}")        # should be a real file path


# ------------------------------------------------------------------------------
# models loads and forward pass
# ------------------------------------------------------------------------------

config = load_config("configs/train_config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model  = load_vgg16(num_classes=4, device=device)
loaders = get_dataloaders(config)

images, labels, paths = next(iter(loaders["train"]))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

print(f"Output shape: {outputs.shape}")  # should be [32, 4]
print(f"Device in use: {device}")



print("All imports OK")
