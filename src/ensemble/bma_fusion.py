"""
Created on May 2026
src/ensemble/bma_fusion.py
------------------------------------------------------------------------------
Ensemble fusion by Bayesian Model Averaging (BMA) — global and per-class.

-----------------------------------------------------------------------
PART 1 — GLOBAL BMA
-----------------------------------------------------------------------
How BMA works:
  In the sequential Bayesian fusion we implemented earlier, one model's output
  becomes the prior and the others update it sequentially. This has two problems:
    1. The result depends on the order of models.
    2. The first model's errors become a biased starting point.

  BMA solves both problems by asking a different question:
    "Instead of treating models as sequential observers, what if I treat the
     choice of model itself as something uncertain?"

  The key idea is:
    - We have 3 models. We don't know which one is the 'correct' model.
    - We assign a probability (weight) to each model: how likely is it that
      THIS model is the best one, given what we've seen on the validation set?
    - At test time, we average each model's predictions, weighted by those
      probabilities.

  The formula is:
      P(class | x) = sum_m [ P(class | x, Model_m) * P(Model_m | val_data) ]

  Where:
    - P(class | x, Model_m)  = the softmax output of model m for image x
                                → this comes from the test set predictions
    - P(Model_m | val_data)  = how much we trust model m overall
                                → this is the BMA weight, computed from val set

  Notice that this looks like weighted soft voting. The critical difference is
  HOW the weights are computed: they come from Bayes' theorem applied to the
  validation data, not from an arbitrary choice like equal weighting.

  Limitation of global BMA:
    A single weight per model assumes the model is either good or bad overall.
    In practice, VGG16 might be better at spalling while ResNet50 is better at
    cracks. A global weight averages over this specialisation and loses it.
    See Part 2 (per-class BMA) for the solution.

Computing the weights - step by step:
  We want P(Model_m | val_data). By Bayes theorem:
      P(Model_m | val_data) proportional to P(val_data | Model_m) * P(Model_m)

  Assuming equal prior over models (P(Model_m) = 1/M for all m), this becomes:
      P(Model_m | val_data) proportional to P(val_data | Model_m)

  P(val_data | Model_m) is the likelihood: how well does model m explain the
  validation labels? We compute it as the product of per-image probabilities
  of the correct class:
      P(val_data | Model_m) = product_i P(y_i | x_i, Model_m)

  Because multiplying many small probabilities causes numerical underflow
  (the result rounds to zero), we work in log space instead:
      log P(val_data | Model_m) = sum_i log P(y_i | x_i, Model_m)

  For each image i we extract the softmax probability the model assigned to
  the TRUE class y_i, take the log, and sum across all validation images.

  To convert back to weights we exponentiate (using the log-sum-exp trick
  for numerical stability) and normalise so they sum to 1.

-----------------------------------------------------------------------
PART 2 — PER-CLASS BMA
-----------------------------------------------------------------------
Motivation:
  Global BMA assigns one weight per model based on overall validation
  performance. But different CNN architectures tend to be better at
  different damage types:
    - VGG16 may recognise spalling well (large texture regions)
    - ResNet50 may recognise cracks well (thin, elongated features)
    - AlexNet may handle undamaged images differently

  Per-class BMA computes a SEPARATE set of weights for each class.
  The weight matrix has shape (num_models, num_classes), and at test time
  each class dimension of the fused output uses a different set of weights.

  Formally, for class c:
      P(class=c | x) proportional to sum_m [ P(class=c | x, Model_m) * weight[m, c] ]

  Where weight[m, c] is derived from how well model m recognised class c
  on the validation set — computed only over validation images whose true
  label IS class c.

  After computing the weighted sum for all classes, we renormalise the
  output so it is a valid probability distribution summing to 1.

  This lets the ensemble exploit model specialisation: if VGG16 is
  particularly good at spalling, it gets a high weight for that class
  even if its overall accuracy is similar to the other models.
"""

import numpy as np


# ==============================================================================
# Shared utility: log-sum-exp normalisation
# ==============================================================================

def _log_likelihoods_to_weights(log_likelihoods: np.ndarray) -> np.ndarray:
    """
    Convert a 1D array of log-likelihoods to normalised weights using the
    log-sum-exp trick.

    The log-sum-exp trick subtracts the maximum value before exponentiating.
    This prevents numerical underflow (very negative log-likelihoods would
    otherwise round to zero after exp()).

    The subtraction cancels out during normalisation, so the result is
    mathematically identical to the naive exp() + normalise approach.

    Parameters
    ----------
    log_likelihoods : np.ndarray, shape (K,)
        Raw log-likelihood values. Can be any scale.

    Returns
    -------
    np.ndarray, shape (K,)
        Normalised weights summing to 1.0.
    """
    # Subtract the maximum before exponentiating to avoid underflow.
    shifted = log_likelihoods - log_likelihoods.max()

    # Exponentiate — now values are in [0, 1] range, largest becomes 1.
    weights_unnorm = np.exp(shifted)

    # Normalise so they sum to 1.
    return weights_unnorm / weights_unnorm.sum()


# ==============================================================================
# PART 1 — Global BMA: one weight per model
# ==============================================================================

def compute_bma_weights(val_probs_arrays: list, val_true_labels: list) -> np.ndarray:
    """
    Compute global BMA model weights from validation set predictions.

    For each model, compute the log-likelihood of the validation data:
        log P(val_data | Model_m) = sum_i log P(y_i | x_i, Model_m)

    Then convert to normalised weights via the log-sum-exp trick.

    Parameters
    ----------
    val_probs_arrays : list of np.ndarray
        One (N_val, num_classes) array per model — softmax probabilities
        for all validation images.
    val_true_labels : list of int
        Ground truth class indices for the N_val validation images.
        Must be in the same order as the rows of each probs array.

    Returns
    -------
    np.ndarray, shape (num_models,)
        Normalised BMA weights — one per model, summing to 1.0.
    """
    true_labels = np.array(val_true_labels, dtype=int)
    num_models  = len(val_probs_arrays)
    row_indices = np.arange(len(true_labels))

    log_likelihoods = np.zeros(num_models)

    for m, probs in enumerate(val_probs_arrays):
        # For each validation image, extract the probability assigned to the
        # TRUE class using advanced numpy indexing.
        # probs[row_indices, true_labels] picks one value per row — the
        # probability of the correct class for that image.
        correct_class_probs = probs[row_indices, true_labels]

        # Clip to avoid log(0) = -inf.
        correct_class_probs = np.clip(correct_class_probs, a_min=1e-10, a_max=None)

        # Sum the log probabilities across all validation images.
        log_likelihoods[m] = np.sum(np.log(correct_class_probs))

    return _log_likelihoods_to_weights(log_likelihoods)


def bma_batch(probs_arrays: list, weights: np.ndarray) -> np.ndarray:
    """
    Apply global BMA fusion across a full dataset (batch of images).

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element is a 2D array of shape (N, num_classes) — one per model.
        These are the TEST set predictions.
    weights : np.ndarray, shape (num_models,)
        Global BMA weights from compute_bma_weights(). Must sum to 1.0.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Weighted average of all models' softmax outputs.
        Each row sums to 1.0.
    """
    # Stack all model arrays along a new axis → shape (num_models, N, num_classes).
    stacked = np.stack(probs_arrays, axis=0).astype(np.float64)

    # Reshape weights to (num_models, 1, 1) so broadcasting applies the
    # correct scalar weight to every image and class for each model.
    w = weights.reshape(-1, 1, 1)

    # Weighted sum along axis=0 (model axis) → shape (N, num_classes).
    return (stacked * w).sum(axis=0)


# ==============================================================================
# PART 2 — Per-class BMA: one weight per (model, class) pair
# ==============================================================================

def compute_bma_weights_per_class(
    val_probs_arrays: list,
    val_true_labels:  list,
    num_classes:      int
) -> np.ndarray:
    """
    Compute per-class BMA weights from validation set predictions.

    For each class c, we only look at validation images whose TRUE label is c,
    and ask: "how well did each model recognise this specific class?"

    This produces a weight matrix of shape (num_models, num_classes) where
    each column is an independent set of weights for one class.

    For class c, model m:
        log_likelihood[m, c] = sum_{i: y_i == c} log P(class=c | x_i, Model_m)

    Then for each class c independently, the column log_likelihood[:, c] is
    converted to normalised weights via the log-sum-exp trick.

    Parameters
    ----------
    val_probs_arrays : list of np.ndarray
        One (N_val, num_classes) array per model.
    val_true_labels : list of int
        Ground truth class indices for the N_val validation images.
    num_classes : int
        Total number of classes (4 in your case).

    Returns
    -------
    np.ndarray, shape (num_models, num_classes)
        Weight matrix. Each COLUMN sums to 1.0.
        weight_matrix[m, c] = how much to trust model m for class c.
    """
    true_labels = np.array(val_true_labels, dtype=int)
    num_models  = len(val_probs_arrays)

    # Initialise the log-likelihood matrix.
    # Shape: (num_models, num_classes)
    # log_likelihoods[m, c] = log-likelihood of model m on class-c images only.
    log_likelihoods = np.zeros((num_models, num_classes))

    for c in range(num_classes):
        # Find the indices of all validation images that truly belong to class c.
        # np.where returns a tuple; [0] gives the 1D array of matching indices.
        class_indices = np.where(true_labels == c)[0]

        if len(class_indices) == 0:
            # No validation images for this class — fall back to equal weights.
            # This should not happen with a balanced dataset but is a safe guard.
            log_likelihoods[:, c] = 0.0
            continue

        for m, probs in enumerate(val_probs_arrays):
            # Extract only the rows belonging to class c.
            # Shape: (N_c, num_classes) where N_c = number of class-c images.
            class_probs = probs[class_indices, :]

            # For these images the correct class IS c, so we extract column c:
            # the probability each model assigned to class c for these images.
            # Shape: (N_c,)
            correct_probs = class_probs[:, c]

            # Clip to avoid log(0).
            correct_probs = np.clip(correct_probs, a_min=1e-10, a_max=None)

            # Sum log probabilities over all class-c validation images.
            log_likelihoods[m, c] = np.sum(np.log(correct_probs))

    # For each class c (each column), independently convert log-likelihoods
    # to normalised weights using the log-sum-exp trick.
    weight_matrix = np.zeros((num_models, num_classes))

    for c in range(num_classes):
        weight_matrix[:, c] = _log_likelihoods_to_weights(log_likelihoods[:, c])

    return weight_matrix


def bma_batch_per_class(probs_arrays: list, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Apply per-class BMA fusion across a full dataset (batch of images).

    For each class c, the fused probability is the weighted sum of each model's
    probability for that class, using the class-specific weights:

        fused[i, c] proportional to sum_m weight_matrix[m, c] * model_m_probs[i, c]

    After computing this for all classes, each row is renormalised to sum to 1
    so the output remains a valid probability distribution.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element is a 2D array of shape (N, num_classes) — one per model.
        These are the TEST set predictions.
    weight_matrix : np.ndarray, shape (num_models, num_classes)
        Per-class weights from compute_bma_weights_per_class().
        Each COLUMN sums to 1.0.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Per-class BMA fused probabilities.
        Each row is renormalised to sum to 1.0.
    """
    # Stack all model outputs → shape (num_models, N, num_classes).
    stacked = np.stack(probs_arrays, axis=0).astype(np.float64)

    # Reshape weight_matrix from (num_models, num_classes)
    # to (num_models, 1, num_classes) so it broadcasts correctly over the N
    # dimension (axis 1). This means:
    #   axis 0 (num_models)  → matched to model axis of stacked
    #   axis 1 (1)           → broadcast over N images
    #   axis 2 (num_classes) → matched to class axis of stacked
    # Each class gets its own weight per model, applied identically to all N
    # images (the weight is a property of the model, not the image).
    w = weight_matrix.reshape(weight_matrix.shape[0], 1, weight_matrix.shape[1])

    # Elementwise multiply and sum over axis=0 (model axis).
    # Result shape: (N, num_classes).
    # Each element fused[i, c] = sum_m weight_matrix[m, c] * probs_m[i, c]
    fused = (stacked * w).sum(axis=0)

    # Renormalise each row so it sums to 1.
    # After per-class weighting the rows no longer automatically sum to 1
    # because each class dimension was scaled by different weights.
    # keepdims=True keeps shape (N, 1) so division broadcasts over num_classes.
    row_sums = fused.sum(axis=1, keepdims=True)

    # Guard against zero-sum rows (should not occur with softmax inputs).
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    return fused / row_sums


# ==============================================================================
# Convenience: compute weights and log them clearly
# ==============================================================================

def compute_and_log_weights(
    val_probs_arrays: list,
    val_true_labels:  list,
    model_names:      list,
    logger
) -> np.ndarray:
    """
    Compute global BMA weights and write them to the logger.

    Returns
    -------
    np.ndarray, shape (num_models,)
    """
    weights = compute_bma_weights(val_probs_arrays, val_true_labels)

    logger.info("  BMA global weights (from validation log-likelihood):")
    for name, w in zip(model_names, weights):
        logger.info(f"    {name:<12}: {w:.4f}")
    logger.info(f"  Weights sum: {weights.sum():.6f}")

    return weights


def compute_and_log_weights_per_class(
    val_probs_arrays: list,
    val_true_labels:  list,
    num_classes:      int,
    model_names:      list,
    class_names:      list,
    logger
) -> np.ndarray:
    """
    Compute per-class BMA weights and write a readable matrix to the logger.

    The logged table shows one row per model and one column per class, so you
    can immediately see which model the framework trusts most for each damage
    type. For example:

        Model           crack    efflorescence    spalling    undamaged
        vgg16           0.2341          0.4102      0.5231       0.1982
        resnet50        0.6812          0.3201      0.3109       0.5843
        alexnet         0.0847          0.2697      0.1660       0.2175

    This shows ResNet50 is most trusted for cracks and undamaged, while
    VGG16 is most trusted for spalling. This is itself a result worth
    reporting in the paper.

    Returns
    -------
    np.ndarray, shape (num_models, num_classes)
    """
    weight_matrix = compute_bma_weights_per_class(
        val_probs_arrays, val_true_labels, num_classes
    )

    col_width = 14
    header = f"  {'Model':<12}" + "".join(f"  {c:>{col_width}}" for c in class_names)
    logger.info("  BMA per-class weights (from validation log-likelihood):")
    logger.info(header)
    logger.info("  " + "-" * (12 + (col_width + 2) * num_classes))

    for m, name in enumerate(model_names):
        row = f"  {name:<12}" + "".join(
            f"  {weight_matrix[m, c]:>{col_width}.4f}" for c in range(num_classes)
        )
        logger.info(row)

    logger.info("  Column sums (should all be 1.0):")
    col_sums = f"  {'':12}" + "".join(
        f"  {weight_matrix[:, c].sum():>{col_width}.4f}" for c in range(num_classes)
    )
    logger.info(col_sums)

    return weight_matrix