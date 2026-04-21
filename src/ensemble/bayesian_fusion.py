"""
Created on Thursday Mar 19 2026
src/ensemble/bayesian_fusion.py
------------------------------------------------------------------------------
Ensemble fusion by Sequential Bayesian Updating.

How sequential Bayesian fusion works:
  Bayesian inference is a framework for updating beliefs (probabilities) as
  new evidence arrives. The core formula is Bayes' theorem:

      P(class | evidence) = P(evidence | class) * P(class) / P(evidence)
      posterior            = likelihood * prior / marginal_likelihood

  In our ensemble context:
    - We treat each model's output as a piece of evidence.
    - We start with the first model's probabilities as our initial belief
      (the prior).
    - We then update this belief sequentially using each subsequent model's
      probabilities as new likelihood evidence.
    - After incorporating all models, the final posterior is our prediction.

  Step by step with 3 models (VGG16 → ResNet50 → AlexNet):

  Step 1 — Initialise prior from VGG16:
      prior = VGG16_probs  e.g. [0.70, 0.10, 0.15, 0.05]

  Step 2 — Update with ResNet50 as likelihood:
      unnormalised_posterior = likelihood * prior (elementwise)
                             = ResNet50_probs * prior
      marginal = sum(unnormalised_posterior)   ← normalisation constant
      posterior_1 = unnormalised_posterior / marginal

  Step 3 — Use posterior_1 as new prior, update with AlexNet:
      unnormalised_posterior = AlexNet_probs * posterior_1
      marginal = sum(unnormalised_posterior)
      final_posterior = unnormalised_posterior / marginal

  The final_posterior is a valid probability distribution (sums to 1.0)
  and represents the combined belief of all three models.

Why this differs from soft voting:
  Soft voting treats all models as independent and averages their outputs.
  Bayesian fusion treats the models as sequential observers — each one
  updates the belief left by the previous one. In theory this is more
  principled because it uses the full chain of evidence. In practice the
  results are often similar to soft voting, with differences emerging when
  models strongly disagree.

Limitation:
  The order of models matters — updating with VGG16 then ResNet50 gives
  a slightly different result than the reverse order. In the current
  implementation we use the fixed order: VGG16 → ResNet50 → AlexNet.
  Averaging over all orderings (6 permutations for 3 models) would give
  a more symmetric result but is more computationally expensive.
"""

import numpy as np


def sequential_bayesian(probs_list: list) -> np.ndarray:
    """
    Apply sequential Bayesian fusion for a single image.

    Parameters
    ----------
    probs_list : list of np.ndarray
        Each element is a 1D array of class probabilities for one model,
        shape (num_classes,). The list is processed in order — the first
        element becomes the initial prior.

    Returns
    -------
    np.ndarray, shape (num_classes,)
        Final posterior probability distribution after incorporating all
        models. Values sum to 1.0.
    """
    # Step 1: Initialise the prior from the first model's probabilities.
    # We make a copy with .copy() so we don't accidentally modify the input.
    prior = np.array(probs_list[0], dtype=np.float64).copy()

    # Step 2: Sequentially update the prior using each subsequent model.
    # We skip index 0 (already used as prior) using slice [1:].
    for likelihood in probs_list[1:]:
        likelihood = np.array(likelihood, dtype=np.float64)

        # Elementwise multiplication: for each class, multiply the current
        # belief (prior) by how likely this model thinks the image is that class.
        unnormalised = likelihood * prior

        # The marginal likelihood is the sum of the unnormalised posterior.
        # Dividing by it normalises the posterior back to a valid probability
        # distribution that sums to 1.0.
        marginal = unnormalised.sum()

        # Guard against division by zero — this can happen if all probabilities
        # are zero for some class (which should not occur with softmax outputs
        # but is a good defensive practice).
        if marginal > 0:
            prior = unnormalised / marginal
        else:
            # If marginal is zero (degenerate case), fall back to uniform distribution.
            prior = np.ones_like(prior) / len(prior)

    return prior


def sequential_bayesian_batch(probs_arrays: list) -> np.ndarray:
    """
    Apply sequential Bayesian fusion across a full dataset (batch of images).

    This is the function called by scripts/ensemble_eval.py and scripts/deploy.py.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element is a 2D array of shape (N, num_classes) — one per model.
        The list is processed in order (index 0 is the initial prior).

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Final posterior distributions for all N images.
    """
    # Step 1: Initialise prior from the first model — shape (N, num_classes).
    prior = probs_arrays[0].astype(np.float64).copy()

    # Step 2: Update sequentially with each subsequent model.
    for likelihood_array in probs_arrays[1:]:
        likelihood = likelihood_array.astype(np.float64)

        # Elementwise multiplication — shape (N, num_classes).
        # Each image's prior is multiplied by the corresponding likelihood row.
        unnormalised = likelihood * prior

        # Sum along axis=1 (class dimension) → shape (N,).
        # keepdims=True keeps the shape as (N, 1) so division broadcasts correctly
        # across all num_classes columns without needing explicit reshaping.
        marginal = unnormalised.sum(axis=1, keepdims=True)

        # np.where(condition, x, y) returns x where condition is True, else y.
        # Here: divide normally where marginal > 0, use uniform where it is 0.
        prior = np.where(
            marginal > 0,
            unnormalised / marginal,
            np.ones_like(unnormalised) / unnormalised.shape[1]
        )

    return prior
