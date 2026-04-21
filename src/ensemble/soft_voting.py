"""
Created on Thursday Mar 19 2026
src/ensemble/soft_voting.py
------------------------------------------------------------------------------
Ensemble fusion by Soft Voting (probability averaging).

How soft voting works:
  Each model produces a probability distribution over all classes for each
  image. Soft voting averages these distributions elementwise. The final
  prediction is the class with the highest average probability.

  Example with 3 models and 4 classes:
    VGG16    → [0.80, 0.10, 0.05, 0.05]   (very confident: crack)
    ResNet50 → [0.60, 0.20, 0.10, 0.10]   (moderately confident: crack)
    AlexNet  → [0.30, 0.40, 0.20, 0.10]   (thinks efflorescence)
    Average  → [0.57, 0.23, 0.12, 0.08]   → "crack" wins

  Compare this to hard voting: hard voting would give 2 votes to crack and
  1 vote to efflorescence, ignoring that VGG16 was 80% sure and AlexNet
  only 40% sure. Soft voting captures this confidence difference.

Strengths:
  - Uses more information than hard voting (full probability distributions,
    not just the argmax).
  - Naturally handles confidence: a highly confident model has more influence
    on the average than an uncertain one.
  - Generally outperforms hard voting in practice.
  - Simple to implement and understand.

Weaknesses:
  - Assumes all models are equally reliable (equal weights).
    Weighted soft voting (giving better models more influence) can improve
    this but requires knowing each model's performance in advance.
  - If models are not well-calibrated (i.e. their probabilities don't
    accurately reflect true confidence), averaging can be misleading.

Weighted vs unweighted:
  This implementation provides both unweighted (equal weights) and weighted
  variants. Weights could be set to each model's validation accuracy,
  for example, so that the best-performing model has more influence.
"""

import numpy as np


def soft_voting(probs_list: list, weights: list = None) -> np.ndarray:
    """
    Apply soft voting (probability averaging) for a single image.

    Parameters
    ----------
    probs_list : list of np.ndarray
        Each element is a 1D array of class probabilities for one model,
        shape (num_classes,). Values should sum to 1.0 (softmax outputs).
    weights : list of float, optional
        Per-model weights for weighted averaging. Must be the same length
        as probs_list and sum to 1.0.
        If None, all models are weighted equally (unweighted average).
        Example: [0.5, 0.3, 0.2] gives 50% weight to the first model.

    Returns
    -------
    np.ndarray, shape (num_classes,)
        Averaged probability distribution. Values sum to 1.0.
        The predicted class is np.argmax of this array.
    """
    num_models  = len(probs_list)
    num_classes = len(probs_list[0])

    if weights is None:
        # Equal weights: each model contributes 1/num_models to the average.
        weights = [1.0 / num_models] * num_models
    else:
        # Validate that weights sum to approximately 1.0.
        # We use a tolerance of 1e-6 to account for floating point rounding.
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {sum(weights):.6f}. "
                f"Normalise your weights before passing them."
            )
        if len(weights) != num_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({num_models})."
            )

    # Compute the weighted sum of probability arrays.
    # We start with zeros and accumulate the weighted contribution of each model.
    averaged = np.zeros(num_classes)
    for prob_array, weight in zip(probs_list, weights):
        # weight * prob_array multiplies each element of the array by the weight.
        # += accumulates the result.
        averaged += weight * np.array(prob_array)

    return averaged


def soft_voting_batch(probs_arrays: list, weights: list = None) -> np.ndarray:
    """
    Apply soft voting across a full dataset (batch of images).

    This is the function called by scripts/ensemble_eval.py and scripts/deploy.py.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element is a 2D array of shape (N, num_classes) — one per model.
    weights : list of float, optional
        Per-model weights. If None, equal weights are used.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Averaged probability distributions for all N images.
    """
    num_models  = len(probs_arrays)
    num_classes = probs_arrays[0].shape[1]

    if weights is None:
        weights = [1.0 / num_models] * num_models

    # Stack all model outputs along a new axis to get shape (num_models, N, num_classes).
    # np.stack(..., axis=0) creates the new axis at position 0.
    stacked = np.stack(probs_arrays, axis=0)

    # Convert weights to a numpy array and reshape to (num_models, 1, 1) so
    # that broadcasting works correctly across the N and num_classes dimensions.
    # Broadcasting means numpy automatically expands the weight array to match
    # the shape of stacked without needing explicit loops.
    w = np.array(weights).reshape(num_models, 1, 1)

    # Weighted sum along axis=0 (the model axis) → shape (N, num_classes).
    averaged = (stacked * w).sum(axis=0)

    return averaged
