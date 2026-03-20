"""
Created on Thursday Mar 19 2026

src/evaluation/metrics.py
------------------------------------------------------------------------------
Metric computation for both individual model evaluation and ensemble comparison.

Metrics computed:
  - Accuracy:    overall fraction of correctly classified images
  - Precision:   of all images predicted as class X, how many actually are X
  - Recall:      of all images that are class X, how many did we correctly find
  - F1 Score:    harmonic mean of precision and recall (balanced)
  - F2 Score:    like F1 but weights recall more heavily than precision.
                 Useful when missing a damage (false negative) is worse than
                 a false alarm (false positive) — which is the case here.
  - Specificity: of all images that are NOT class X, how many did we correctly
                 identify as not-X. Also called True Negative Rate.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score, f1_score
from src.utils.logger import get_logger

# For printing results
logger = get_logger(__name__)


# ==============================================================================
# Main metric computation
# ==============================================================================

def calculate_metrics(predictions: list, true_labels: list, class_names: list) -> dict:
    """
    This function computes both overall (weighted average) metrics and
    per-class metrics in a single call, returning everything in a
    structured dictionary that can be directly saved to JSON.

    The 'weighted' average means each class contributes to the overall
    metric proportionally to how many samples it has. This is appropriate
    when classes are imbalanced.

    Parameters
    ----------
    predictions : list of int
        Predicted class indices, e.g. [0, 2, 1, 3, 0, ...].
    true_labels : list of int
        Ground truth class indices, same length as predictions.
    class_names : list of str
        Class names in label-index order, e.g. ["crack", "efflorescence",
        "spalling", "undamaged"]. Index 0 = "crack", index 1 = "efflorescence"
        and so on.

    Returns
    -------
    dict with structure:
        {
            "overall": {
                "accuracy":    float,
                "precision":   float,
                "recall":      float,
                "f1":          float,
                "f2":          float,
            },
            "per_class": {
                "crack": {
                    "precision":   float,
                    "recall":      float,
                    "f1":          float,
                    "specificity": float,
                },
                "efflorescence": { ... },
                ...
            }
        }
    """
    num_classes = len(class_names)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(num_classes))
    )

    # ------------------------------------------------------------------
    # Overall metrics (weighted average across all classes)
    # ------------------------------------------------------------------
    # Accuracy (TP+TN)/(TP+FP+TN+FN)
    accuracy  = accuracy_score(true_labels, predictions)

    # zero_division=0 means: if a class has no predicted samples (precision
    # is undefined), return 0 instead of raising a warning.
    # Precision (TP)/(TP+FP)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)

    # Recall (TP)/(TP+FN)
    recall    = recall_score(true_labels, predictions, average="weighted", zero_division=0)

    # F1 Score 2*(Precision*Recall)/(Precision+Recall)
    f1        = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    # F2 Score
    # Formula: (1 + beta²) * (precision * recall) / (beta² * precision + recall)
    beta = 2
    if (beta**2 * precision + recall) > 0:
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    else:
        f2 = 0.0

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------
    # precision_score with average=None returns one value per class.
    per_class_precision = precision_score(
        true_labels, predictions, average=None,
        labels=list(range(num_classes)), zero_division=0
    )
    per_class_recall = recall_score(
        true_labels, predictions, average=None,
        labels=list(range(num_classes)), zero_division=0
    )
    per_class_f1 = f1_score(
        true_labels, predictions, average=None,
        labels=list(range(num_classes)), zero_division=0
    )

    # Specificity per class = TN / (TN + FP)
    # For class i:
    #   TN = sum of all cm values EXCEPT row i and column i
    #   FP = sum of column i EXCEPT the diagonal (cm[i][i])
    # A simpler way: specificity[i] = cm[i][i] / sum(cm[:, i])
    # which is "of all predictions for class i, how many were correct"
    # Note: this is actually the same as precision. The correct formulation
    # for specificity uses the full confusion matrix:
    per_class_specificity = []
    for i in range(num_classes):
        # True negatives: all correct predictions that are not class i
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        # False positives: images of other classes predicted as class i
        fp = cm[:, i].sum() - cm[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_specificity.append(float(specificity_i))

    # ------------------------------------------------------------------
    # Assemble results dictionary
    # ------------------------------------------------------------------
    per_class_dict = {}
    for i, name in enumerate(class_names):
        per_class_dict[name] = {
            "precision":   round(float(per_class_precision[i]), 4),
            "recall":      round(float(per_class_recall[i]),    4),
            "f1":          round(float(per_class_f1[i]),        4),
            "specificity": round(per_class_specificity[i],      4),
        }

    results = {
        "overall": {
            "accuracy":  round(float(accuracy),  4),
            "precision": round(float(precision), 4),
            "recall":    round(float(recall),    4),
            "f1":        round(float(f1),        4),
            "f2":        round(float(f2),        4),
        },
        "per_class": per_class_dict,
    }

    # ------------------------------------------------------------------
    # Log a readable summary
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Overall Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  F2 Score:  {f2:.4f}")
    logger.info("Per-class Metrics:")
    for name, m in per_class_dict.items():
        logger.info(
            f"  {name:<16} | P: {m['precision']:.4f} | "
            f"R: {m['recall']:.4f} | F1: {m['f1']:.4f} | "
            f"Spec: {m['specificity']:.4f}"
        )
    logger.info("=" * 50)

    return results


# ==============================================================================
# Comparison utility
# ==============================================================================

def build_comparison_table(method_metrics: dict) -> list:
    """
    Build a flat list of per-method summary rows suitable for saving to CSV
    or displaying as a table.

    This is used by scripts/evaluate.py after running all ensemble methods
    to produce a side-by-side comparison.

    Parameters
    ----------
    method_metrics : dict
        Keys are method names (e.g. "hard_voting"), values are the metric
        dicts returned by calculate_metrics().

    Returns
    -------
    list of dict
        Each dict has keys: Method, Accuracy, Precision, Recall, F1, F2.
        One row per method — ready to pass to pd.DataFrame() or save_json().

    Example output:
        [
            {"Method": "hard_voting",  "Accuracy": 0.87, "F1": 0.86, ...},
            {"Method": "soft_voting",  "Accuracy": 0.89, "F1": 0.88, ...},
            {"Method": "bayesian",     "Accuracy": 0.88, "F1": 0.87, ...},
        ]
    """
    rows = []
    for method_name, metrics in method_metrics.items():
        overall = metrics["overall"]
        rows.append({
            "Method":    method_name,
            "Accuracy":  overall["accuracy"],
            "Precision": overall["precision"],
            "Recall":    overall["recall"],
            "F1":        overall["f1"],
            "F2":        overall["f2"],
        })
    return rows
