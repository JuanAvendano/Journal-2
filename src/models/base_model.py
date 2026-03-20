"""
Created on Thursday Mar 19 2026

src/models/base_model.py
------------------------------------------------------------------------------
Shared training and evaluation logic for all CNN models.

Previously, VGG16.py, ResNet50.py, and AlexNet.py each contained their own
copies of the training loop, validation loop, and CSV saving functions. Since
the logic was identical across all three, it has been extracted here into a
single shared module.

How this works with the individual model files:
  - vgg16.py, resnet50.py, alexnet.py each define ONLY their architecture
    loader function (e.g. load_vgg16).
  - scripts/train.py calls the architecture loader to get the model, then
    passes it to train_model() and evaluate_final_model() from this file.
  - None of the three model files need to know about training loops.

This pattern is called "separation of concerns" — each file has one clear
responsibility, making the codebase easier to read, test, and modify.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.utils.io_utils import save_predictions_csv
from src.utils.logger import get_logger

# Module-level logger. When this module logs a message, it will be tagged
# with "src.models.base_model" so you can trace where it came from.
logger = get_logger(__name__)


# ==============================================================================
# Checkpoint saving and loading
# ==============================================================================

def save_checkpoint(model: nn.Module, save_dir: Path, filename: str) -> None:
    """
    Save the model's learned weights to a .pth file.

    We save only the state_dict (the weights), not the entire model object.
    This is the recommended PyTorch practice because:
      - It is more portable — you can load the weights into any model with
        the same architecture, regardless of how the model class was defined.
      - It produces smaller files than saving the full model object.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model.
    save_dir : Path
        Directory to save into (e.g. saved_models/vgg16/).
    filename : str
        File name, either "best.pth" or "last.pth".
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    # model.state_dict() returns an OrderedDict of parameter name → tensor.
    torch.save(model.state_dict(), save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load previously saved weights into a model.

    The model must have the same architecture as when the weights were saved —
    you cannot load VGG16 weights into a ResNet50.

    Parameters
    ----------
    model : nn.Module
        An initialised model (architecture already built, weights not yet loaded).
    checkpoint_path : Path
        Path to the .pth file produced by save_checkpoint().
    device : torch.device
        Where to map the loaded weights (CPU or CUDA GPU).

    Returns
    -------
    nn.Module
        The same model object, now with the loaded weights.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Have you trained this model yet?"
        )

    # map_location ensures the weights are loaded onto the correct device
    # even if they were saved on a different device (e.g. saved on GPU,
    # loaded on CPU). Without this you can get a CUDA device mismatch error.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Checkpoint loaded: {checkpoint_path}")

    return model


# ==============================================================================
# Validation loop (used during training each epoch)
# ==============================================================================

def evaluate_validation_set(
    model: nn.Module,
    val_loader,
    device: torch.device
) -> tuple:
    """
    Run the model on the full validation set and collect outputs.

    This function is called at the end of every training epoch to measure
    how well the model generalises to unseen data. No weight updates happen
    here — gradients are disabled for efficiency.

    Parameters
    ----------
    model : nn.Module
        The model in its current training state.
    val_loader : DataLoader
        DataLoader for the validation set (uses ImageFolderWithPaths).
    device : torch.device
        GPU or CPU.

    Returns
    -------
    tuple:
        all_outputs  : torch.Tensor, shape (N, num_classes) — raw logits
        all_labels   : torch.Tensor, shape (N,)             — true labels
        all_probs    : np.ndarray,   shape (N, num_classes) — softmax probs
        predictions  : list of int                          — predicted class indices
        filenames    : list of str                          — image basenames
    """
    # model.eval() switches off training-specific behaviour:
    #   - Dropout layers stop randomly zeroing activations.
    #   - BatchNorm layers use running statistics instead of batch statistics.
    # Always call model.eval() before running inference.
    model.eval()

    all_outputs  = []
    all_labels   = []
    all_probs    = []
    predictions  = []
    filenames    = []

    # torch.no_grad() disables gradient computation entirely.
    # During training, PyTorch builds a computational graph to enable
    # backpropagation. We don't need that during validation, so disabling
    # it saves memory and speeds up the forward pass.
    with torch.no_grad():
        for inputs, labels, paths in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass — run the images through the model.
            outputs = model(inputs)

            # Collect raw outputs (logits) and true labels.
            # .cpu() moves the tensor from GPU back to CPU memory before
            # appending — we accumulate on CPU to avoid filling GPU memory.
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

            # torch.softmax converts raw logits to probabilities.
            # dim=1 means we apply softmax across the class dimension,
            # so each row (image) sums to 1.0.
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            # torch.max returns (values, indices). We only want the indices
            # (the predicted class), so we use _ to discard the values.
            _, predicted = torch.max(outputs, 1)
            predictions += predicted.cpu().tolist()

            # os.path.basename extracts just the filename from a full path,
            # e.g. "C:/data/train/crack/img001.jpg" → "img001.jpg"
            filenames += [os.path.basename(p) for p in paths]

    # torch.cat concatenates a list of tensors along dimension 0 (the batch
    # dimension), giving us one tensor covering the full validation set.
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_labels_tensor  = torch.cat(all_labels,  dim=0)

    # np.concatenate does the same for numpy arrays.
    all_probs_array = np.concatenate(all_probs, axis=0)

    return all_outputs_tensor, all_labels_tensor, all_probs_array, predictions, filenames


# ==============================================================================
# Helper evaluation functions
# ==============================================================================

def evaluate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy from model outputs and true labels.

    Parameters
    ----------
    outputs : torch.Tensor, shape (N, num_classes)
        Raw logits from the model (before softmax).
    labels : torch.Tensor, shape (N,)
        True class indices.

    Returns
    -------
    float
        Accuracy in range [0.0, 1.0].
    """
    # torch.max along dim=1 returns the index of the highest logit per row,
    # which is the predicted class.
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total   = labels.size(0)
    return correct / total


def evaluate_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module
) -> float:
    """
    Compute the loss value from model outputs and true labels.

    Parameters
    ----------
    outputs : torch.Tensor, shape (N, num_classes)
        Raw logits from the model.
    labels : torch.Tensor, shape (N,)
        True class indices.
    criterion : nn.Module
        The loss function (e.g. CrossEntropyLoss).

    Returns
    -------
    float
        Scalar loss value.
    """
    loss = criterion(outputs, labels)
    # .item() extracts the scalar value from a single-element tensor.
    return loss.item()


# ==============================================================================
# Main training loop
# ==============================================================================

def train_model(
    model: nn.Module,
    model_name: str,
    loaders: dict,
    criterion: nn.Module,
    optimizer,
    config: dict,
    run_dir: Path
) -> dict:
    """
    Train the model for a given number of epochs and save the best checkpoint.

    This function implements the full training loop:
      1. For each epoch, run all training batches (forward pass, loss,
         backward pass, weight update).
      2. At the end of each epoch, evaluate on the validation set.
      3. If validation accuracy improved, save the model as best.pth and
         save the validation predictions CSV.
      4. Always save the model as last.pth at the end of each epoch.
      5. If early stopping is enabled and no improvement is seen for
         'patience' epochs, stop training early.

    Parameters
    ----------
    model : nn.Module
        The model to train (already loaded with pretrained weights).
    model_name : str
        Name string used for saving files, e.g. "vgg16".
    loaders : dict
        Dictionary with keys "train" and "val", each a DataLoader.
    criterion : nn.Module
        Loss function, typically nn.CrossEntropyLoss().
    optimizer : torch.optim.Optimizer
        Optimiser, typically optim.Adam().
    config : dict
        Full training config from train_config.yaml.
    run_dir : Path
        Timestamped directory where all outputs for this run are saved.

    Returns
    -------
    dict
        Training history with keys:
          "train_losses", "val_losses", "train_accs", "val_accs"
        Each is a list with one value per epoch. Used by plots.py to draw
        the learning curves.
    """
    # Read settings from config
    num_epochs = config["training"]["epochs"]
    save_dir   = Path(config["paths"]["saved_models"]) / model_name

    # Early stopping settings
    es_cfg      = config.get("early_stopping", {})
    es_enabled  = es_cfg.get("enabled", True)
    es_patience = es_cfg.get("patience", 10)

    class_names = config["dataset"]["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")

    # ------------------------------------------------------------------
    # Tracking variables
    # ------------------------------------------------------------------
    best_val_accuracy  = 0.0   # Track the best validation accuracy seen so far
    epochs_no_improve  = 0     # Counter for early stopping

    # History lists — one entry appended per epoch.
    # These are returned at the end and used to draw learning curves.
    history = {
        "train_losses": [],
        "val_losses":   [],
        "train_accs":   [],
        "val_accs":     [],
    }

    logger.info(f"Starting training: {model_name} | {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ==============================================================
        # Training phase
        # ==============================================================
        # model.train() re-enables dropout and batch norm training mode
        # (the opposite of model.eval() called during validation).
        model.train()

        running_loss    = 0.0
        correct_train   = 0
        total_train     = 0

        for inputs, labels, _ in loaders["train"]:
            # The _ discards the file paths — we don't need them during training.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients from the previous batch.
            # PyTorch accumulates gradients by default, so we must clear them
            # at the start of each batch to prevent them from building up.
            optimizer.zero_grad()

            # Forward pass: compute the model's predictions.
            outputs = model(inputs)

            # Compute the loss between predictions and true labels.
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients of the loss with respect to
            # every model parameter. This is the core of backpropagation.
            loss.backward()

            # Update the model parameters using the computed gradients.
            optimizer.step()

            # Accumulate statistics for this batch.
            running_loss  += loss.item()
            _, predicted   = torch.max(outputs, 1)
            total_train   += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Average loss and accuracy over all batches in this epoch.
        train_loss = running_loss / len(loaders["train"])
        train_acc  = correct_train / total_train

        # ==============================================================
        # Validation phase
        # ==============================================================
        val_outputs, val_labels, val_probs, val_preds, val_files = \
            evaluate_validation_set(model, loaders["val"], device)

        val_acc  = evaluate_accuracy(val_outputs, val_labels)
        val_loss = evaluate_loss(val_outputs, val_labels, criterion)

        epoch_time = time.time() - epoch_start

        # Log the epoch summary in the same format as your original code.
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] {epoch_time:.0f}s — "
            f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        # Append to history for learning curve plots.
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        # ==============================================================
        # Save best checkpoint
        # ==============================================================
        if val_acc > best_val_accuracy:
            logger.info(f"  Validation accuracy improved "
                        f"({best_val_accuracy:.4f} → {val_acc:.4f}). Saving best.pth")
            best_val_accuracy = val_acc

            epochs_no_improve = 0   # Reset the early stopping counter

            # Save best weights
            save_checkpoint(model, save_dir, "best.pth")

            # Save the validation predictions CSV for this best epoch.
            # This is the CSV that the ensemble methods will later read.
            preds_dir = run_dir / "predictions"
            save_predictions_csv(
                save_dir   = preds_dir,
                filename   = "predictions.csv",
                image_names   = val_files,
                true_labels   = val_labels.tolist(),
                predicted_labels = val_preds,
                class_probs   = val_probs,
                class_names   = class_names
            )
        else:
            epochs_no_improve += 1

        # Always save the most recent weights as last.pth, overwriting
        # the previous one. This lets you resume training from the last
        # epoch if needed, even if it wasn't the best epoch.
        save_checkpoint(model, save_dir, "last.pth")

        # ==============================================================
        # Early stopping check
        # ==============================================================
        if es_enabled and epochs_no_improve >= es_patience:
            logger.info(
                f"Early stopping triggered after {epoch+1} epochs "
                f"({es_patience} epochs without improvement)."
            )
            break

    logger.info(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    return history


# ==============================================================================
# Final evaluation on the test set
# ==============================================================================

def evaluate_final_model(
    model: nn.Module,
    model_name: str,
    test_loader,
    criterion: nn.Module,
    config: dict,
    run_dir: Path
) -> tuple:
    """
    Evaluate the best saved model on the held-out test set.

    This is called after training completes. It loads the best checkpoint,
    runs inference on the test set, saves the predictions CSV, and returns
    the predictions and labels for the evaluation module to compute metrics.

    Parameters
    ----------
    model : nn.Module
        The model (architecture already built).
    model_name : str
        e.g. "vgg16" — used to locate the best.pth checkpoint.
    test_loader : DataLoader
        DataLoader for the test set.
    criterion : nn.Module
        Loss function (used to report test loss).
    config : dict
        Full training config.
    run_dir : Path
        Timestamped run directory for saving outputs.

    Returns
    -------
    tuple:
        predictions : list of int   — predicted class indices
        labels_list : list of int   — true class indices
        all_probs   : np.ndarray    — softmax probabilities (N, num_classes)
        filenames   : list of str   — image basenames
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best checkpoint saved during training.
    save_dir       = Path(config["paths"]["saved_models"]) / model_name
    checkpoint_path = save_dir / "best.pth"
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    class_names = config["dataset"]["class_names"]

    running_loss = 0.0
    correct      = 0
    total        = 0
    predictions  = []
    labels_list  = []
    all_probs    = []
    filenames    = []

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            labels_list += labels.tolist()
            filenames   += [os.path.basename(p) for p in paths]

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            loss          = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy().tolist())

    test_loss = running_loss / len(test_loader)
    test_acc  = correct / total
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    all_probs_array = np.concatenate(all_probs, axis=0)

    # Save test set predictions CSV alongside the validation predictions.
    preds_dir = run_dir / "predictions"
    save_predictions_csv(
        save_dir         = preds_dir,
        filename         = "test_predictions.csv",
        image_names      = filenames,
        true_labels      = labels_list,
        predicted_labels = predictions,
        class_probs      = all_probs_array,
        class_names      = class_names
    )

    return predictions, labels_list, all_probs_array, filenames
