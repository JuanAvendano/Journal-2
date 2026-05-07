"""
src/ensemble/svm_meta_learner.py
------------------------------------------------------------------------------
Ensemble fusion by SVM Meta-Learner (Stacking).

Background — why SVM as a meta-learner?
  Like the MLP meta-learner, the SVM meta-learner is a stacking approach:
  a second-level model is trained to learn how to combine the outputs of the
  base CNN classifiers. The difference is in HOW the combination is learned.

  An SVM (Support Vector Machine) finds the decision boundary that maximises
  the margin between classes. In a feature space of stacked model probabilities,
  this means the SVM learns which combinations of model outputs most reliably
  separate the four damage categories.

  Why compare SVM to MLP for this task?
    - The meta-learner is trained on a small dataset (validation set predictions
      only — typically a few hundred samples). SVMs are known to generalise well
      on small datasets because they focus on the support vectors (the most
      informative samples near the boundary) and ignore the rest.
    - SVMs have fewer hyperparameters than an MLP, which reduces the risk of
      overfitting on small training sets.
    - By comparing SVM and MLP, the paper can empirically address whether a
      non-linear learned combiner (MLP) offers any advantage over a
      kernel-based combiner (SVM) for this specific stacking task.

The kernel choice:
  We use an RBF (Radial Basis Function) kernel, also called a Gaussian kernel:

      K(x, z) = exp(-gamma * ||x - z||^2)

  The RBF kernel maps the input features into an infinite-dimensional space,
  allowing the SVM to learn non-linear boundaries in the original feature space.
  This is appropriate because the relationship between stacked CNN probabilities
  and the correct class is unlikely to be linearly separable.

  Alternative: a linear kernel is also sensible here since the input features
  (probabilities) are already in [0, 1] and soft voting (a form of linear
  combination) already performs reasonably well. You can test both by changing
  the kernel parameter in the config or function call.

Multi-class strategy:
  SVMs are inherently binary classifiers. For 4 classes we use the
  One-vs-Rest (OvR) strategy: train one SVM per class that separates
  "this class vs all others". The class with the highest decision score wins.

  sklearn's SVC with decision_function_shape='ovr' handles this automatically.

Probability outputs:
  Standard SVMs output a decision score, not a probability. We enable
  probability=True in sklearn's SVC, which uses Platt scaling (a logistic
  regression fitted on the decision scores via cross-validation) to convert
  scores to calibrated probabilities. This allows the SVM to integrate
  seamlessly with the rest of the pipeline, which expects probability arrays.

  Important: Platt scaling adds a cross-validation step during training, so
  training with probability=True is slightly slower than without.

The feature vector:
  Identical to the MLP meta-learner. For each image we concatenate the
  4-class probability vectors from all base models:
    [vgg16_crack, vgg16_efflo, vgg16_spal, vgg16_undam,
     resnet_crack, resnet_efflo, resnet_spal, resnet_undam,
     alex_crack,  alex_efflo,  alex_spal,  alex_undam]  → 12 features for 3 models
                                                          → 20 features for 5 models

Data leakage note:
  Identical concern as the MLP: the SVM must be trained on VALIDATION set
  predictions, not training set predictions. The base models were trained on
  the training set, so their outputs on training images are overfit and
  would give the SVM an inflated, unrealistic signal.

Saving and loading:
  sklearn models are saved with joblib, which serialises Python objects to
  disk. This is the standard approach for sklearn models (as opposed to
  PyTorch's .pth format used for the MLP).
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Feature construction (shared with MLP — identical logic)
# ==============================================================================

def build_meta_features(probs_arrays: list) -> np.ndarray:
    """
    Stack model probability arrays into a single feature matrix.

    This is the same function used in the MLP meta-learner. It is duplicated
    here so that svm_meta_learner.py is self-contained and does not create
    a circular import with mlp_meta_learner.py.

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.
        Example: 3 arrays of shape (N, 4) → feature matrix of shape (N, 12).

    Returns
    -------
    np.ndarray, shape (N, num_models * num_classes)
        Feature matrix where each row is the concatenated probability vectors
        from all models for one image.
    """
    # np.hstack joins arrays side by side along the column axis.
    # [[a, b], [c, d]] and [[e, f], [g, h]] → [[a, b, e, f], [c, d, g, h]]
    return np.hstack(probs_arrays)


# ==============================================================================
# Training the SVM meta-learner
# ==============================================================================

def train_svm(
    probs_arrays: list,
    true_labels: list,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    save_path: Path = None,
) -> Pipeline:
    """
    Train the SVM meta-learner on stacked base model outputs.

    This function is called by scripts/ensemble_eval.py after loading the
    validation set predictions from all base models, exactly like train_mlp().

    Parameters
    ----------
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.
        These MUST be validation set predictions to avoid data leakage.
    true_labels : list of int
        Ground truth class indices for the N images.
    kernel : str
        SVM kernel type. Options: "rbf" (default), "linear", "poly".
        - "rbf"    : Radial Basis Function — good default for non-linear problems.
        - "linear" : Linear boundary — fast, interpretable, sometimes competitive.
    C : float
        Regularisation parameter. Controls the trade-off between a wide margin
        and correct classification of training points.
        - Larger C: smaller margin, fewer training errors (risk of overfitting).
        - Smaller C: wider margin, more training errors (more regularisation).
        Default 1.0 is a reasonable starting point.
    gamma : str or float
        Kernel coefficient for "rbf" and "poly" kernels.
        - "scale" (default): 1 / (n_features * X.var()) — adapts to feature scale.
        - "auto"           : 1 / n_features.
        - float            : Manual value.
    save_path : Path, optional
        If provided, saves the trained SVM pipeline (scaler + SVM) to this path
        using joblib. Use a .pkl or .joblib extension.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A trained Pipeline containing [StandardScaler → SVC].
        Call pipeline.predict_proba(X) to get probability outputs.

    Notes on StandardScaler:
        Even though CNN probabilities are already in [0, 1], scaling is still
        beneficial because the SVM's RBF kernel uses Euclidean distances.
        If some features have slightly different variance (e.g., one model is
        always very confident while another is uncertain), unscaled features
        would give disproportionate weight to the high-variance ones.
        StandardScaler normalises each feature to zero mean and unit variance,
        ensuring all probability dimensions contribute equally to the kernel.

        We use a Pipeline (scaler + SVM) so that the same scaler fitted on
        training features is automatically applied to test features at inference
        time. This prevents a common bug where you scale training but forget
        to apply the same transform to the test set.
    """
    # Build the stacked feature matrix from all model probability arrays.
    X = build_meta_features(probs_arrays)   # shape (N, num_models * num_classes)
    y = np.array(true_labels)               # shape (N,)

    logger.info("Training SVM meta-learner:")
    logger.info(f"  Input features : {X.shape[1]} ({len(probs_arrays)} models × "
                f"{X.shape[1] // len(probs_arrays)} classes)")
    logger.info(f"  Training samples: {len(y)}")
    logger.info(f"  Kernel: {kernel}  |  C: {C}  |  gamma: {gamma}")

    # ------------------------------------------------------------------
    # Build the pipeline: StandardScaler → SVC
    # ------------------------------------------------------------------
    # A Pipeline chains preprocessing and modelling into a single object.
    # When you call pipeline.fit(X, y), it runs:
    #   1. scaler.fit_transform(X)   ← fit and transform training data
    #   2. svc.fit(scaled_X, y)      ← train the SVM on scaled features
    # When you call pipeline.predict_proba(X_test), it runs:
    #   1. scaler.transform(X_test)  ← apply the SAME scaler (fitted on train)
    #   2. svc.predict_proba(scaled_X_test)
    # This ensures consistent scaling without manual bookkeeping.

    pipeline = Pipeline([
        (
            # Step 1: Standardise features to zero mean and unit variance.
            "scaler",
            StandardScaler()
        ),
        (
            # Step 2: SVM classifier.
            # probability=True enables Platt scaling so we can call
            # predict_proba() and get calibrated probability outputs.
            # decision_function_shape='ovr' uses One-vs-Rest for multi-class.
            "svc",
            SVC(
                kernel                 = kernel,
                C                      = C,
                gamma                  = gamma,
                probability            = True,
                decision_function_shape= "ovr",
                random_state           = 42,   # reproducibility
            )
        ),
    ])

    # Fit the full pipeline on training data.
    pipeline.fit(X, y)

    logger.info("SVM meta-learner training complete.")

    # Save the fitted pipeline to disk if a path is given.
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        logger.info(f"SVM pipeline saved: {save_path}")

    return pipeline


# ==============================================================================
# Inference with the trained SVM
# ==============================================================================

def svm_predict(pipeline: Pipeline, probs_arrays: list) -> np.ndarray:
    """
    Run inference with a trained SVM meta-learner pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A trained Pipeline returned by train_svm() or loaded by load_svm().
    probs_arrays : list of np.ndarray
        Each array has shape (N, num_classes) — one per base model.
        These should be the TEST set predictions for fair evaluation.

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Calibrated probability distributions for all N images.
        Row i sums to 1.0. argmax(axis=1) gives the predicted class index.
    """
    # Build the same stacked feature matrix used during training.
    X = build_meta_features(probs_arrays)

    # predict_proba() applies the scaler then returns SVM probabilities.
    # Shape: (N, num_classes).
    probs = pipeline.predict_proba(X)

    return probs


# ==============================================================================
# Loading a saved SVM pipeline
# ==============================================================================

def load_svm(save_path: Path) -> Pipeline:
    """
    Load a previously saved SVM pipeline from disk.

    Parameters
    ----------
    save_path : Path
        Path to the .pkl / .joblib file saved by train_svm().

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded pipeline, ready for inference via svm_predict().
    """
    pipeline = joblib.load(save_path)
    logger.info(f"SVM meta-learner loaded from: {save_path}")
    return pipeline
