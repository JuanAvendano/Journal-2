"""
scripts/experiments/bayesian_permutations.py
------------------------------------------------------------------------------
Experiment: find the best model ordering for Sequential Bayesian Fusion.

Background:
  Sequential Bayesian fusion is order-dependent — the first model in the
  sequence sets the initial prior, and each subsequent model updates it.
  Different orderings can produce different final posteriors and therefore
  different classification accuracy.

  With 3 models (VGG16, ResNet50, AlexNet) there are 3! = 6 possible
  orderings. This experiment runs all 6, computes metrics for each, and
  compares them against the other ensemble methods (hard voting, soft voting)
  so you can see both:
    (a) Which Bayesian ordering is best.
    (b) Whether the best Bayesian ordering outperforms the non-Bayesian methods.

Usage:
    python scripts/experiments/bayesian_permutations.py
    python scripts/experiments/bayesian_permutations.py --show_plots

No retraining is needed — this experiment reads the existing prediction CSVs
produced during training.

Output:
    results/experiments/bayesian_permutations/<timestamp>/
        permutation_summary.json      ← all 6 orderings + metrics
        permutation_summary.csv       ← same, easier to read in Excel
        full_comparison.json          ← all 6 orderings + voting methods
        full_comparison.csv
        best_order.json               ← winner ordering and its metrics
        confusion_matrices/
            vgg16_resnet50_alexnet.png
            vgg16_alexnet_resnet50.png
            ... (one per ordering)
            hard_voting.png
            soft_voting.png
        plots/
            bayesian_orderings_comparison.png  ← bar chart, 6 orderings
            full_comparison.png                ← bar chart, all methods
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from itertools import permutations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.io_utils   import (load_config, make_run_dir, save_json,
                                   load_predictions_csv, save_metrics_csv,
                                   resolve_prediction_path)
from src.utils.logger     import get_run_logger

from src.ensemble.bayesian_fusion import sequential_bayesian_batch
from src.ensemble.hard_voting     import hard_voting_batch
from src.ensemble.soft_voting     import soft_voting_batch

from src.evaluation.metrics          import calculate_metrics, build_comparison_table
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.plots            import plot_metric_comparison


# ==============================================================================
# Configuration — edit paths here if needed
# ==============================================================================

ENSEMBLE_CONFIG = "configs/ensemble_config.yaml"
TRAIN_RESULTS   = "results/training"   # Where per-model prediction CSVs live


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all Bayesian model orderings and compare with "
                    "voting methods."
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively in addition to saving them."
    )
    return parser.parse_args()


# ==============================================================================
# Plotting helper
# ==============================================================================

def plot_ordering_comparison(
    rows: list,
    title: str,
    filename: str,
    save_dir: Path,
    highlight_methods: list = None,
    show: bool = False
) -> None:
    """
    Plot a grouped bar chart comparing multiple methods/orderings.

    This wraps plot_metric_comparison with a custom title and optional
    visual highlighting of the non-Bayesian baseline methods.

    Parameters
    ----------
    rows : list of dict
        As returned by build_comparison_table().
    title : str
        Chart title.
    filename : str
        Output PNG filename.
    save_dir : Path
        Directory to save into.
    highlight_methods : list of str, optional
        Method names to visually distinguish (e.g. voting baselines).
        Currently used only to note in the title — full visual highlighting
        would require a custom plot function.
    show : bool
        Whether to display interactively.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as mticker

    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).set_index("Method")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "F2"]

    fig, ax = plt.subplots(figsize=(max(11, len(rows) * 1.5), 6))

    df[metrics_to_plot].T.plot(
        kind="bar",
        ax=ax,
        edgecolor="white",
        width=0.75
    )

    ax.set_xlabel("Metric",  fontsize=12)
    ax.set_ylabel("Score",   fontsize=12)
    ax.set_title(title,      fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.xticks(rotation=0, fontsize=11)
    ax.legend(title="Method / Ordering", fontsize=9, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()

    path = save_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args   = parse_args()
    config = load_config(ENSEMBLE_CONFIG)

    class_names = config["dataset"]["class_names"]
    num_classes = config["dataset"]["num_classes"]

    # ------------------------------------------------------------------
    # Create experiment run directory
    # ------------------------------------------------------------------
    run_dir = make_run_dir(
        base_dir             = "results/experiments/bayesian_permutations",
        model_or_method_name = "run"
    )
    logger = get_run_logger(run_dir, __name__)
    logger.info("Bayesian Permutations Experiment")
    logger.info(f"Run directory: {run_dir}")

    # ------------------------------------------------------------------
    # Load test predictions for all models
    # ------------------------------------------------------------------
    logger.info("Loading test predictions...")
    model_probs = {}   # model_name → np.ndarray (N, num_classes)
    true_labels = None
    image_names = None

    model_names = list(config["model_predictions"].keys())

    for model_name in model_names:
        path_str = config["model_predictions"][model_name]
        try:
            resolved = resolve_prediction_path(
                path_str   = path_str,
                base_dir   = TRAIN_RESULTS,
                model_name = model_name
            )
            csv_path = resolved.parent / "test_predictions.csv"

            if not csv_path.exists():
                logger.warning(f"  Test predictions not found for "
                               f"{model_name}: {csv_path}")
                continue

            data = load_predictions_csv(csv_path, class_names)
            model_probs[model_name] = data["probs"]

            # Capture true labels and image names from the first model.
            # All models must have the same images in the same order.
            if true_labels is None:
                true_labels = data["true_labels"]
                image_names = data["image_names"]

            logger.info(f"  Loaded {model_name}: {len(data['true_labels'])} images")

        except FileNotFoundError as e:
            logger.warning(f"  Could not load {model_name}: {e}")

    if len(model_probs) < 2:
        logger.error("Need at least 2 models. Exiting.")
        return

    available_models = list(model_probs.keys())
    logger.info(f"Models available: {available_models}")
    logger.info(f"Test images: {len(true_labels)}")

    # ------------------------------------------------------------------
    # Run all permutations of the Bayesian ordering
    # ------------------------------------------------------------------
    # itertools.permutations(list) generates all possible orderings.
    # For 3 models: 3! = 6 orderings.
    # For 2 models: 2! = 2 orderings.
    all_orderings = list(permutations(available_models))
    logger.info(f"\nRunning {len(all_orderings)} Bayesian orderings...")

    bayesian_metrics = {}   # ordering_label → metrics dict
    cm_dir           = run_dir / "confusion_matrices"

    for ordering in all_orderings:
        # Create a human-readable label for this ordering.
        # e.g. ("vgg16", "resnet50", "alexnet") → "vgg16 → resnet50 → alexnet"
        label      = " → ".join(ordering)
        # Filename-safe version (no spaces or arrows).
        label_safe = "_".join(ordering)

        logger.info(f"  Ordering: {label}")

        # Build the probs_arrays list in this specific order.
        ordered_probs = [model_probs[m] for m in ordering]

        # Run sequential Bayesian fusion.
        fused_probs = sequential_bayesian_batch(ordered_probs)
        predictions = np.argmax(fused_probs, axis=1).tolist()

        # Compute metrics.
        metrics = calculate_metrics(predictions, true_labels, class_names)
        metrics["ordering"] = label
        metrics["method"]   = f"bayesian_{label_safe}"

        bayesian_metrics[label] = metrics

        logger.info(f"    Accuracy: {metrics['overall']['accuracy']:.4f} | "
                    f"F1: {metrics['overall']['f1']:.4f}")

        # Save individual confusion matrix.
        plot_confusion_matrix(
            true_labels = true_labels,
            predictions = predictions,
            class_names = class_names,
            title       = f"Bayesian: {label}",
            save_dir    = cm_dir,
            filename    = f"bayesian_{label_safe}.png",
            show        = args.show_plots
        )

        # Save per-ordering metrics JSON.
        save_json(metrics, run_dir / "metrics" / f"bayesian_{label_safe}.json")

    # ------------------------------------------------------------------
    # Run baseline methods (hard voting, soft voting)
    # ------------------------------------------------------------------
    logger.info("\nRunning baseline methods...")

    # Use all available models for the baselines.
    all_probs = [model_probs[m] for m in available_models]

    baseline_metrics = {}

    for method_name, fusion_fn in [("hard_voting", hard_voting_batch),
                                    ("soft_voting",  soft_voting_batch)]:
        fused_probs = fusion_fn(all_probs)
        predictions = np.argmax(fused_probs, axis=1).tolist()
        metrics     = calculate_metrics(predictions, true_labels, class_names)
        metrics["method"] = method_name

        baseline_metrics[method_name] = metrics
        logger.info(f"  {method_name}: "
                    f"Accuracy={metrics['overall']['accuracy']:.4f} | "
                    f"F1={metrics['overall']['f1']:.4f}")

        plot_confusion_matrix(
            true_labels = true_labels,
            predictions = predictions,
            class_names = class_names,
            title       = method_name.replace("_", " ").title(),
            save_dir    = cm_dir,
            filename    = f"{method_name}.png",
            show        = args.show_plots
        )

        save_json(metrics, run_dir / "metrics" / f"{method_name}.json")

    # ------------------------------------------------------------------
    # Find the best Bayesian ordering
    # ------------------------------------------------------------------
    # Sort orderings by F1 score (descending) to find the winner.
    best_label = max(
        bayesian_metrics,
        key=lambda k: bayesian_metrics[k]["overall"]["f1"]
    )
    best_metrics = bayesian_metrics[best_label]

    logger.info(f"\nBest Bayesian ordering: {best_label}")
    logger.info(f"  Accuracy: {best_metrics['overall']['accuracy']:.4f}")
    logger.info(f"  F1:       {best_metrics['overall']['f1']:.4f}")

    save_json(
        {"best_ordering": best_label, "metrics": best_metrics},
        run_dir / "best_order.json"
    )

    # ------------------------------------------------------------------
    # Build comparison tables and save summaries
    # ------------------------------------------------------------------
    # Table 1: Bayesian orderings only.
    # Rename keys to use the "method" field so build_comparison_table works.
    bayesian_for_table = {
        v["method"]: v for v in bayesian_metrics.values()
    }
    bayesian_rows = build_comparison_table(bayesian_for_table)

    # Sort rows by F1 descending so the best ordering appears first.
    bayesian_rows.sort(key=lambda r: r["F1"], reverse=True)

    save_json(
        {"orderings": bayesian_rows},
        run_dir / "permutation_summary.json"
    )
    save_metrics_csv(
        save_dir     = run_dir,
        filename     = "permutation_summary.csv",
        metrics_list = bayesian_rows
    )

    # Table 2: All methods combined (Bayesian orderings + voting baselines).
    all_methods_for_table = {**bayesian_for_table, **baseline_metrics}
    all_rows = build_comparison_table(all_methods_for_table)
    all_rows.sort(key=lambda r: r["F1"], reverse=True)

    save_json(
        {"methods": all_rows},
        run_dir / "full_comparison.json"
    )
    save_metrics_csv(
        save_dir     = run_dir,
        filename     = "full_comparison.csv",
        metrics_list = all_rows
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plots_dir = run_dir / "plots"

    # Plot 1: Bayesian orderings only.
    # Rename Method labels to be shorter for readability in the chart.
    # e.g. "bayesian_vgg16_resnet50_alexnet" → "vgg16→resnet50→alexnet"
    bayesian_rows_display = []
    for row in bayesian_rows:
        display_row = dict(row)
        # Reconstruct the arrow label from the ordering stored in the metrics.
        ordering_label = bayesian_metrics[
            next(k for k, v in bayesian_metrics.items()
                 if v["method"] == row["Method"])
        ]["ordering"]
        display_row["Method"] = ordering_label
        bayesian_rows_display.append(display_row)

    plot_ordering_comparison(
        rows     = bayesian_rows_display,
        title    = "Bayesian Fusion — All Model Orderings",
        filename = "bayesian_orderings_comparison.png",
        save_dir = plots_dir,
        show     = args.show_plots
    )

    # Plot 2: All methods combined.
    # Make baseline labels more readable.
    all_rows_display = []
    for row in all_rows:
        display_row = dict(row)
        method = row["Method"]
        if method.startswith("bayesian_"):
            # Find the arrow label for this ordering.
            ordering_label = bayesian_metrics[
                next(k for k, v in bayesian_metrics.items()
                     if v["method"] == method)
            ]["ordering"]
            display_row["Method"] = f"Bayes: {ordering_label}"
        else:
            display_row["Method"] = method.replace("_", " ").title()
        all_rows_display.append(display_row)

    plot_ordering_comparison(
        rows              = all_rows_display,
        title             = "Full Comparison: Bayesian Orderings vs Voting Methods",
        filename          = "full_comparison.png",
        save_dir          = plots_dir,
        highlight_methods = ["Hard Voting", "Soft Voting"],
        show              = args.show_plots
    )

    # ------------------------------------------------------------------
    # Final summary printed to log
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE — Results ranked by F1 (descending)")
    logger.info("=" * 70)
    logger.info(f"{'Method/Ordering':<40} {'Accuracy':>10} {'F1':>10}")
    logger.info("-" * 70)
    for row in all_rows_display:
        logger.info(
            f"{row['Method']:<40} "
            f"{row['Accuracy']:>10.4f} "
            f"{row['F1']:>10.4f}"
        )
    logger.info("=" * 70)
    logger.info(f"Best Bayesian ordering: {best_label}")
    logger.info(f"Results saved to: {run_dir}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()
