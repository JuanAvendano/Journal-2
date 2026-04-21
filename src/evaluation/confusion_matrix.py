"""
Created on Thursday Mar 19 2026

src/evaluation/confusion_matrix.py
------------------------------------------------------------------------------
Confusion matrix plotting and saving.

This module separates the display logic (plotting) from saving logic, and
supports both showing the matrix interactively and saving it to a PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix as confusion_matrix

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    true_labels: list,
    predictions: list,
    class_names: list,
    title: str,
    save_dir: Path = None,
    filename: str = "confusion_matrix.png",
    show: bool = False
) -> None:
    """
    Plot and optionally save a confusion matrix heatmap.

    Both the raw counts and the normalised fractions are displayed in each
    cell to see both the absolute numbers and the proportions.

    Parameters
    ----------
    true_labels : list of int
        Ground truth class indices.
    predictions : list of int
        Predicted class indices.
    class_names : list of str
        Class names in label-index order.
    title : str
        Title displayed above the matrix, e.g. "VGG16" or "Soft Voting".
    save_dir : Path, optional
        If provided, the figure is saved as a PNG in this directory.
        If None, the figure is only displayed (or neither if show=False).
    filename : str
        Filename for the saved PNG. Default "confusion_matrix.png".
    show : bool
        If True, call plt.show() to display the figure interactively.
        Set to False when running training in batch/script mode to avoid
        blocking execution waiting for the window to be closed.
    """
    num_classes = len(class_names)
    labels      = list(range(num_classes))

    # Compute the raw confusion matrix (integer counts).
    cm_raw = confusion_matrix(true_labels, predictions, labels=labels)

    # Normalise each row by dividing by the row sum.
    # cm_raw.sum(axis=1, keepdims=True) gives a column vector of row sums.
    # Dividing broadcasts correctly across all columns.
    # np.where avoids division by zero for any class with no samples.
    row_sums   = cm_raw.sum(axis=1, keepdims=True)
    cm_norm    = np.where(row_sums > 0, cm_raw / row_sums, 0.0)

    # ------------------------------------------------------------------
    # Build annotation strings: show both fraction and raw count
    # ------------------------------------------------------------------
    # Each cell displays e.g. "0.92\n(46)" meaning 92% of that true class
    # was predicted as this column's class, and that is 46 images.
    annot = np.empty_like(cm_raw, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annot[i, j] = f"{cm_norm[i, j]:.2f}\n({cm_raw[i, j]})"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # sns.heatmap draws a coloured grid where higher values are darker.
    # annot=annot uses our custom annotation strings (not the default numbers).
    # fmt="" is required when using a custom annotation array (not a number).
    # cmap="Blues" uses a blue colour scale — high values (correct predictions)
    # appear dark blue, low values appear white.
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        linewidths=0.5,      # thin lines between cells for readability
        linecolor="lightgrey"
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix: {title}", fontsize=14, fontweight="bold")

    # Rotate tick labels for readability if class names are long.
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0,  fontsize=10)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save and/or display
    # ------------------------------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        # dpi=150 gives a reasonably high-resolution PNG suitable for reports.
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {save_path}")

    if show:
        plt.show()

    # Always close the figure after saving/showing to free memory.
    # If you forget this, matplotlib accumulates open figures and eventually
    # raises a warning or runs out of memory during long training runs.
    plt.close(fig)


def plot_confusion_matrix_grid(
    all_metrics: dict,
    class_names: list,
    save_dir: Path = None,
    show: bool = False
) -> None:
    """
    Plot multiple confusion matrices side by side in a grid layout.

    Useful for the ensemble comparison in scripts/ensemble_eval.py — you can
    see all fusion methods' confusion matrices at a glance.

    Parameters
    ----------
    all_metrics : dict
        Keys are method names, values are dicts with keys "true_labels"
        and "predictions" (lists of int).
        Example: {"hard_voting": {"true_labels": [...], "predictions": [...]}}
    class_names : list of str
        Class names in label-index order.
    save_dir : Path, optional
        Where to save the combined figure.
    show : bool
        Whether to display interactively.
    """
    methods     = list(all_metrics.keys())
    num_methods = len(methods)

    # Arrange subplots in a row — one per method.
    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))

    # If there is only one method, axes is a single Axes object, not a list.
    # Wrap it in a list for uniform handling below.
    if num_methods == 1:
        axes = [axes]

    num_classes = len(class_names)
    labels      = list(range(num_classes))

    for ax, method in zip(axes, methods):
        true_labels = all_metrics[method]["true_labels"]
        predictions = all_metrics[method]["predictions"]

        cm_raw  = confusion_matrix(true_labels, predictions, labels=labels)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm_raw / row_sums, 0.0)

        annot = np.empty_like(cm_raw, dtype=object)
        for i in range(num_classes):
            for j in range(num_classes):
                annot[i, j] = f"{cm_norm[i, j]:.2f}\n({cm_raw[i, j]})"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            linecolor="lightgrey",
            cbar=False          # hide the colour bar for individual subplots
        )
        ax.set_title(method.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=9)

    fig.suptitle("Confusion Matrices — Ensemble Method Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "confusion_matrices_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison confusion matrix saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
