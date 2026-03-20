"""
src/ensemble/sugeno_fuzzy.py
------------------------------------------------------------------------------
Ensemble fusion by the Sugeno Fuzzy Integral.

Background — what is the Sugeno fuzzy integral?
  The Sugeno fuzzy integral is a method from fuzzy set theory for aggregating
  multiple sources of information (here: model outputs) when those sources
  may have different levels of importance and may not be independent.

  Unlike soft voting (which assumes equal, independent models) and Bayesian
  fusion (which assumes conditional independence), the Sugeno integral uses
  a "fuzzy measure" to capture the importance and interaction between subsets
  of models. This makes it theoretically more expressive, though it requires
  learning or estimating the fuzzy measure from data.

The fuzzy measure (density):
  The fuzzy measure assigns an importance value to every possible subset of
  models, not just to individual models. For 3 models {M1, M2, M3}, it
  assigns values to:
    {M1}, {M2}, {M3}          ← individual importances
    {M1,M2}, {M1,M3}, {M2,M3} ← pair importances
    {M1,M2,M3}                ← full set (always = 1.0 by definition)

  In the simplest version (used here), we only specify the individual
  model densities and derive the rest using the lambda-fuzzy measure rule,
  which introduces a single parameter λ that captures interaction effects.

How the Sugeno integral is computed for one class:
  1. Sort the model confidence scores for that class in descending order.
  2. For each position in the sorted list, compute the fuzzy measure of the
     set containing the top-i models.
  3. The Sugeno integral is: max over i of min(score_i, fuzzy_measure_i).

  This "max-min" operation is the defining characteristic of the Sugeno
  integral. It finds the best supported level of evidence across all
  possible subsets of models.

References:
  Sugeno, M. (1974). Theory of fuzzy integrals and its applications.
  Grabisch, M. (1995). Fuzzy integral in multicriteria decision making.
"""

import numpy as np
from itertools import combinations


def compute_lambda(densities: list) -> float:
    """
    Compute the lambda parameter for the lambda-fuzzy measure.

    Lambda is a scalar that captures the overall interaction between models.
    It is the unique real solution to the equation:

        1 + lambda = product over all i of (1 + lambda * density_i)

    Lambda > 0: models are complementary (their combination is more valuable
                than the sum of their individual contributions).
    Lambda = 0: models are additive (independent, no interaction).
    Lambda < 0: models are redundant (their combination adds less value than
                the sum of their individual contributions).

    The equation is solved iteratively using the bisection method — a simple
    numerical root-finding technique that works by repeatedly halving an
    interval known to contain the root.

    Parameters
    ----------
    densities : list of float
        Individual importance values for each model. Each value in [0, 1].
        Must not all sum to exactly 1.0 (that would make lambda = 0 trivially).

    Returns
    -------
    float
        The lambda parameter.
    """
    # If densities sum to 1, lambda is 0 (additive measure, equivalent to
    # soft voting with the densities as weights).
    total = sum(densities)
    if abs(total - 1.0) < 1e-9:
        return 0.0

    # Define the equation we want to solve: f(lambda) = 0
    # f(lambda) = product(1 + lambda * d_i) - (1 + lambda)
    def equation(lam):
        product = 1.0
        for d in densities:
            product *= (1.0 + lam * d)
        return product - (1.0 + lam)

    # Bisection method: search in [-1 + epsilon, 100].
    # The root is guaranteed to exist in this range for valid density values.
    lo, hi = -1.0 + 1e-9, 100.0

    # Safety check: if both endpoints have the same sign, the root is not
    # in this interval. This should not happen with valid inputs.
    if equation(lo) * equation(hi) > 0:
        return 0.0

    # Iterate up to 200 times, halving the interval each time.
    # After 200 iterations the interval is smaller than 10^-60, which is
    # more than sufficient precision.
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if equation(mid) == 0.0 or (hi - lo) / 2.0 < 1e-12:
            return mid
        if equation(lo) * equation(mid) < 0:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


def compute_fuzzy_measure(subset_indices: tuple, densities: list, lam: float) -> float:
    """
    Compute the fuzzy measure of a subset of models using the lambda rule.

    For a subset S, the fuzzy measure is:
        g(S) = (1/lambda) * [product over i in S of (1 + lambda * density_i) - 1]

    For lambda = 0 (additive case):
        g(S) = sum of densities in S

    Parameters
    ----------
    subset_indices : tuple of int
        Indices of the models in this subset, e.g. (0, 1) for {M1, M2}.
    densities : list of float
        Individual model importance values.
    lam : float
        Lambda parameter computed by compute_lambda().

    Returns
    -------
    float
        Fuzzy measure value in [0, 1].
    """
    if abs(lam) < 1e-9:
        # Additive (lambda ≈ 0): fuzzy measure is just the sum of densities.
        return sum(densities[i] for i in subset_indices)

    product = 1.0
    for i in subset_indices:
        product *= (1.0 + lam * densities[i])

    return (product - 1.0) / lam


def sugeno_integral_single_class(scores: list, densities: list) -> float:
    """
    Compute the Sugeno fuzzy integral for one class across all models.

    Parameters
    ----------
    scores : list of float
        Each model's probability/confidence for this class.
        e.g. [0.8, 0.6, 0.3] for three models.
    densities : list of float
        Individual importance (fuzzy density) for each model.

    Returns
    -------
    float
        The Sugeno integral value for this class. Higher means more
        evidence from the ensemble that the image belongs to this class.
    """
    num_models = len(scores)
    lam        = compute_lambda(densities)

    # Step 1: Sort models by their score for this class in descending order.
    # argsort returns the indices that would sort the array in ascending order.
    # [::-1] reverses it to descending order.
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores  = [scores[i] for i in sorted_indices]

    # Step 2: For each prefix of the sorted list, compute min(score, g(subset)).
    # The prefix at position i contains the top-(i+1) scoring models.
    result = 0.0
    for i in range(num_models):
        # The subset contains the top-(i+1) models (indices 0 through i in
        # the sorted order, mapped back to original model indices).
        subset  = tuple(sorted_indices[:i+1])
        g_value = compute_fuzzy_measure(subset, densities, lam)

        # Sugeno integral: take the max over all i of min(score_i, g(subset_i))
        value  = min(sorted_scores[i], g_value)
        result = max(result, value)

    return result


def sugeno_fuzzy(probs_list: list, densities: list = None) -> np.ndarray:
    """
    Apply Sugeno fuzzy integral fusion for a single image.

    Parameters
    ----------
    probs_list : list of np.ndarray
        Each element is a 1D array of class probabilities for one model,
        shape (num_classes,).
    densities : list of float, optional
        Individual importance values for each model. Must be in [0, 1].
        If None, equal densities are used (1/num_models each).
        These can be set to each model's validation accuracy to give
        better-performing models more influence.

    Returns
    -------
    np.ndarray, shape (num_classes,)
        Sugeno integral values for each class. Note: these do NOT sum to
        1.0 (the Sugeno integral is not a probability distribution). The
        predicted class is still obtained via argmax.
    """
    num_models  = len(probs_list)
    num_classes = len(probs_list[0])

    if densities is None:
        # Default: equal importance for all models.
        densities = [1.0 / num_models] * num_models

    # Validate densities.
    if len(densities) != num_models:
        raise ValueError(
            f"Number of densities ({len(densities)}) must match "
            f"number of models ({num_models})."
        )
    if any(d < 0 or d > 1 for d in densities):
        raise ValueError("All density values must be in the range [0, 1].")

    # Compute the Sugeno integral independently for each class.
    # For each class, we look at how confident each model is and combine
    # those scores using the fuzzy integral.
    result = np.zeros(num_classes)
    for c in range(num_classes):
        # Extract each model's confidence for class c.
        class_scores = [float(probs[c]) for probs in probs_list]
        result[c]    = sugeno_integral_single_class(class_scores, densities)

    return result


def sugeno_fuzzy_batch(probs_arrays: list, densities: list = None) -> np.ndarray:
    """
    Apply Sugeno fuzzy integral fusion across a full dataset.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element has shape (N, num_classes) — one per model.
    densities : list of float, optional
        Per-model importance values. If None, equal densities are used.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Sugeno integral values for all N images.
    """
    num_images = probs_arrays[0].shape[0]

    results = []
    for i in range(num_images):
        # Extract the probability vector for image i from each model.
        per_image = [arr[i] for arr in probs_arrays]
        results.append(sugeno_fuzzy(per_image, densities))

    return np.vstack(results)
