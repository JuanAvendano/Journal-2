"""
src/ensemble/mlp_meta_learner.py
------------------------------------------------------------------------------
Ensemble fusion by MLP Meta-Learner (Stacking).

Background — what is stacking?
  Stacking (short for stacked generalisation) is an ensemble technique where
  a second-level model (the "meta-learner") learns how to best combine the
  outputs of the first-level models (the "base learners").

  Unlike the other fusion methods in this repository (which use fixed rules
  like averaging or Bayesian updating), the MLP meta-learner is TRAINED.
  It learns from data which combination of model outputs leads to the
  correct prediction.

  The workflow is:
    1. Train VGG16, ResNet50, and AlexNet on the training set (already done).
    2. Collect their probability outputs on a held-out set (the validation set
       predictions CSV files saved during training).
    3. Stack (concatenate) these outputs into a feature vector per image.
    4. Train a small MLP to predict the correct class from this feature vector.
    5. At inference time, run the three base models and pass their outputs
       to the trained MLP to get the final prediction.

The feature vector:
  For each image we concatenate the 4-class probability vectors from all
  three models into a single feature vector of length 12 (3 models × 4 classes).
  Example for one image:
    [vgg16_crack, vgg16_efflo, vgg16_spal, vgg16_undam,   ← VGG16 probs
     resnet_crack, resnet_efflo, resnet_spal, resnet_undam, ← ResNet50 probs
     alex_crack, alex_efflo, alex_spal, alex_undam]          ← AlexNet probs

Why use an MLP as the meta-learner?
  An MLP can learn non-linear relationships between model outputs. For example,
  it might learn that "when VGG16 is uncertain but ResNet50 is confident, trust
  ResNet50" — a pattern that soft voting cannot capture because it weights all
  models equally regardless of their uncertainty.

Important note on data leakage:
  The MLP must NOT be trained on the same data used to train the base models.
  If it were, the base models would have memorised the training data and their
  outputs would be artificially good, making the MLP overfit.
  The correct approach (implemented here) is to train the MLP on the VALIDATION
  set predictions, which the base models have never been trained on.
  Ideally, k-fold cross-validation would be used to generate out-of-fold
  predictions for training the meta-learner.

Architecture:
  Input:   12 features (3 models × 4 classes)
  Hidden1: 64 neurons + ReLU + Dropout(0.3)
  Hidden2: 32 neurons + ReLU + Dropout(0.3)
  Output:  4 neurons (one per class) + softmax
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# MLP architecture
# ==============================================================================

class MLPMetaLearner(nn.Module):
    """
    A small multi-layer perceptron that takes stacked model probabilities
    as input and outputs class probabilities.

    This class inherits from nn.Module, which is the base class for all
    PyTorch neural networks. The two methods we must implement are:
      __init__: define the layers
      forward:  define how data flows through the layers
    """

    def __init__(self, input_size: int, num_classes: int,
                 hidden_sizes: list = None, dropout: float = 0.3):
        """
        Parameters
        ----------
        input_size : int
            Number of input features. For 3 models and 4 classes: 3 × 4 = 12.
        num_classes : int
            Number of output classes (4 for damage detection).
        hidden_sizes : list of int, optional
            Number of neurons in each hidden layer. Default [64, 32].
        dropout : float
            Dropout probability. A fraction of neurons are randomly set to
            zero during training to prevent overfitting.
        """
        # Always call super().__init__() first when subclassing nn.Module.
        super(MLPMetaLearner, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # ------------------------------------------------------------------
        # Build layers dynamically based on hidden_sizes
        # ------------------------------------------------------------------
        # nn.Sequential wraps a sequence of layers into a single module.
        # Data flows through them in order.
        layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            # nn.Linear(in, out) is a fully connected layer:
            #   output = input @ weight.T + bias
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalisation stabilises training by normalising the
            # activations of each layer. Particularly helpful for small datasets.
            layers.append(nn.BatchNorm1d(hidden_size))

            # ReLU (Rectified Linear Unit): f(x) = max(0, x).
            # This introduces non-linearity — without it, stacking linear
            # layers would be equivalent to a single linear layer.
            layers.append(nn.ReLU())

            # Dropout randomly zeros out neurons during training.
            # This forces the network to learn redundant representations
            # and prevents it from relying too heavily on any single neuron.
            layers.append(nn.Dropout(p=dropout))

            prev_size = hidden_size

        # Final output layer — no activation here because CrossEntropyLoss
        # applies softmax internally.
        layers.append(nn.Linear(prev_size, num_classes))

        # Assign the sequential block as an attribute so PyTorch tracks it.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: data flows from input through all layers to output.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_size)
            Stacked model probabilities for a batch of images.

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Raw logits (before softmax).
        """
        return self.network(x)


# ==============================================================================
# Training the meta-learner
# ==============================================================================

def build_meta_features(probs_arrays: list) -> np.ndarray:
    """
    Stack model probability arrays into a single feature matrix.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.

    Returns
    -------
    np.ndarray, shape (N, num_models * num_classes)
        Feature matrix where each row is the concatenated probability
        vectors from all models for one image.
    """
    # np.hstack concatenates arrays horizontally (along columns).
    # For 3 arrays of shape (N, 4), the result has shape (N, 12).
    return np.hstack(probs_arrays)


def train_mlp(
    probs_arrays: list,
    true_labels: list,
    num_classes: int,
    hidden_sizes: list = None,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: Path = None,
    device: torch.device = None
) -> MLPMetaLearner:
    """
    Train the MLP meta-learner on stacked base model outputs.

    This function is called by scripts/ensemble_eval.py after loading the
    validation set predictions from all three base models.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.
        These should be the VALIDATION set predictions (never training set).
    true_labels : list of int
        Ground truth class indices for the N images.
    num_classes : int
        Number of damage classes (4).
    hidden_sizes : list of int, optional
        Hidden layer sizes. Default [64, 32].
    dropout : float
        Dropout probability.
    learning_rate : float
        Adam learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for MLP training.
    save_path : Path, optional
        If provided, saves the trained MLP weights to this path.
    device : torch.device, optional
        If None, automatically uses GPU if available.

    Returns
    -------
    MLPMetaLearner
        The trained MLP model, ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the stacked feature matrix.
    X = build_meta_features(probs_arrays)   # shape (N, num_models * num_classes)
    y = np.array(true_labels)               # shape (N,)

    input_size = X.shape[1]   # 12 for 3 models × 4 classes

    logger.info(f"Training MLP meta-learner:")
    logger.info(f"  Input features: {input_size}")
    logger.info(f"  Training samples: {len(y)}")
    logger.info(f"  Hidden layers: {hidden_sizes or [64, 32]}")
    logger.info(f"  Epochs: {epochs}")

    # Convert numpy arrays to PyTorch tensors.
    # torch.FloatTensor is the standard float32 tensor type for model inputs.
    # torch.LongTensor is required for class labels in CrossEntropyLoss.
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # TensorDataset wraps tensors into a dataset that DataLoader can iterate.
    dataset    = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate and move the MLP to the device.
    model = MLPMetaLearner(
        input_size   = input_size,
        num_classes  = num_classes,
        hidden_sizes = hidden_sizes,
        dropout      = dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop.
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log every 10 epochs to avoid cluttering the output.
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    logger.info("MLP meta-learner training complete.")

    # Save weights if a path is provided.
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"MLP weights saved: {save_path}")

    return model


# ==============================================================================
# Inference with the trained meta-learner
# ==============================================================================

def mlp_predict(
    model: MLPMetaLearner,
    probs_arrays: list,
    device: torch.device = None
) -> np.ndarray:
    """
    Run inference with a trained MLP meta-learner on a full dataset.

    Parameters
    ----------
    model : MLPMetaLearner
        A trained MLP (from train_mlp() or loaded from a saved checkpoint).
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.
    device : torch.device, optional
        If None, uses GPU if available.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Softmax probability distributions for all N images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X        = build_meta_features(probs_arrays)
    X_tensor = torch.FloatTensor(X).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        # Apply softmax to convert logits to probabilities.
        # dim=1 applies softmax across the class dimension.
        probs  = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()


def load_mlp(save_path: Path, input_size: int, num_classes: int,
             hidden_sizes: list = None, device: torch.device = None) -> MLPMetaLearner:
    """
    Load a previously saved MLP meta-learner from a checkpoint file.

    Parameters
    ----------
    save_path : Path
        Path to the saved .pth file.
    input_size : int
        Must match the input size used during training (num_models × num_classes).
    num_classes : int
        Number of output classes.
    hidden_sizes : list of int, optional
        Must match the architecture used during training.
    device : torch.device, optional

    Returns
    -------
    MLPMetaLearner
        The loaded model, ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPMetaLearner(
        input_size   = input_size,
        num_classes  = num_classes,
        hidden_sizes = hidden_sizes
    )
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"MLP meta-learner loaded from: {save_path}")
    return model
