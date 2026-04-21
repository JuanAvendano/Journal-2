"""
Created on Thursday Mar 19 2026
src/ensemble/hard_voting.py
------------------------------------------------------------------------------
Ensemble fusion by Hard Voting.

How hard voting works:
  Each model independently predicts the most likely class for an image.
  The final prediction is whichever class received the most votes across
  all models. This is the simplest possible ensemble method — it treats
  each model as a single voter with one vote.

  Example with 3 models:
    VGG16    → predicts "crack"         (vote 1)
    ResNet50 → predicts "crack"         (vote 2)
    AlexNet  → predicts "efflorescence" (vote 3)
    Result   → "crack" wins (2 votes vs 1)

Strengths:
  - Simple and interpretable — easy to explain what the method does.
  - Robust to one model being confidently wrong, as long as the majority
    is correct.

Weaknesses:
  - Ignores how confident each model is. A model that is 51% sure of its
    prediction gets the same weight as one that is 99% sure.
  - With an even number of models, ties are possible (handled by taking
    the first tied class — see implementation below).
  - With 3 models and 4 classes, a model can "win" with just 1 vote if
    all three models predict differently.

Comparison with soft voting:
  Hard voting only uses the argmax (predicted class) from each model.
  Soft voting uses the full probability distributions. Soft voting
  generally outperforms hard voting because it uses more information.
"""

import numpy as np


def hard_voting(probs_list: list) -> np.ndarray:
    """
    Apply hard voting across a list of model probability arrays.

    Parameters
    ----------
    probs_list : list of np.ndarray
        Each element is a 1D array of class probabilities for one model,
        shape (num_classes,). The list has one entry per model.
        Example: [vgg16_probs, resnet50_probs, alexnet_probs]
        where each is e.g. [0.7, 0.1, 0.1, 0.1]

    Returns
    -------
    np.ndarray, shape (num_classes,)
        A one-hot vector where the winning class has value 1 and all
        others have value 0.
        Example: [1, 0, 0, 0] means class 0 ("crack") won the vote.

    Notes
    -----
    The output is a one-hot vector rather than a raw vote count so that
    the format is consistent with the other fusion methods, all of which
    return a probability-like array that argmax can be applied to.
    """
    num_classes = len(probs_list[0])

    # Step 1: Each model votes for its highest-probability class.
    # np.argmax returns the index of the maximum value in the array.
    # For probs = [0.1, 0.7, 0.1, 0.1], argmax = 1 ("efflorescence").
    votes = [int(np.argmax(probs)) for probs in probs_list]

    # Step 2: Count votes per class and find the winner.
    # max(set(votes), key=votes.count) finds the value in votes that
    # appears most often.
    # If there is a tie (e.g. votes = [0, 1, 2]), this returns the first
    # tied value encountered — which is deterministic but arbitrary.
    winning_class = max(set(votes), key=votes.count)

    # Step 3: Build a one-hot output vector.
    # np.zeros creates an array of zeros, then we set the winning class to 1.
    result = np.zeros(num_classes)
    result[winning_class] = 1

    return result


def hard_voting_batch(probs_arrays: list) -> np.ndarray:
    """
    Apply hard voting across a full dataset (batch of images).

    This is the function called by scripts/ensemble_eval.py and scripts/deploy.py.
    It processes all images at once rather than one at a time.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each element is a 2D array of shape (N, num_classes) — one per model.
        N is the number of images in the dataset.
        Example: [vgg16_probs, resnet50_probs, alexnet_probs]
        where each has shape (N, 4).

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        One-hot encoded predictions for all N images.
    """
    # zip(*probs_arrays) transposes the list-of-arrays structure so that
    # we iterate over images rather than models.
    # For 3 models and N images, zip produces N tuples, each containing
    # the 3 models' probability vectors for that image.
    results = [
        hard_voting(list(per_image_probs))
        for per_image_probs in zip(*probs_arrays)
    ]

    # Stack the list of 1D arrays into a single 2D array (N, num_classes).
    return np.vstack(results)
