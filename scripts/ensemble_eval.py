"""
Created on Thursday Mar 19 2026
scripts/ensemble_eval.py
------------------------------------------------------------------------------
Entry point for ensemble evaluation.

This script loads the per-model prediction CSV files produced during training,
runs all configured fusion methods, computes metrics for each, and saves a
full set of comparison outputs (metrics JSON, confusion matrices, bar charts).

It also handles training and evaluating the MLP meta-learner, which requires
a slightly different workflow from the other methods because it needs to be
trained before it can make predictions.

Usage:
    python scripts/ensemble_eval.py --config configs/ensemble_config.yaml

    # To also display plots interactively:
    python scripts/ensemble_eval.py --config configs/ensemble_config.yaml --show_plots

    # To include the MLP meta-learner (trains it on the fly):
    python scripts/ensemble_eval.py --config configs/ensemble_config.yaml --train_mlp

Output structure (one timestamped folder per run):
    results/ensemble/2026-03-19_15-00/
        metrics/
            hard_voting.json
            soft_voting.json
            bayesian_fusion.json
            sugeno_fuzzy.json
            mlp_meta_learner.json     ← only if --train_mlp is passed
            comparison_summary.json   ← all methods side by side
            comparison_summary.csv    ← same data, CSV format
        confusion_matrices/
            hard_voting.png
            soft_voting.png
            bayesian_fusion.png
            sugeno_fuzzy.png
            confusion_matrices_comparison.png  ← all methods in one figure
        plots/
            ensemble_metric_comparison.png
            per_class_f1_comparison.png
        run.log

Notes on the MLP meta-learner:
    The MLP is trained on the VALIDATION set predictions (the predictions.csv
    files from training). This is important to avoid data leakage — the base
    models were trained on the training set, so using training set predictions
    to train the MLP would give it artificially inflated performance because
    the base models have memorised those images.

    The evaluation of the MLP (computing its metrics) is done on the TEST set
    predictions (test_predictions.csv), same as all other methods.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path for "from src.xxx" imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils   import load_config, make_run_dir, save_json, load_predictions_csv, save_metrics_csv, resolve_prediction_path
from src.utils.logger     import get_run_logger

from src.ensemble.hard_voting      import hard_voting_batch
from src.ensemble.soft_voting      import soft_voting_batch
from src.ensemble.bayesian_fusion  import sequential_bayesian_batch
from src.ensemble.sugeno_fuzzy     import sugeno_fuzzy_batch
from src.ensemble.mlp_meta_learner import train_mlp, mlp_predict

from src.evaluation.metrics          import calculate_metrics, build_comparison_table
from src.evaluation.confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_grid
from src.evaluation.plots            import plot_metric_comparison, plot_per_class_comparison


# ==============================================================================
# Ensemble method registry
# ==============================================================================
# Maps method name strings (from ensemble_config.yaml) to their batch
# functions. Adding a new method only requires adding one entry here.
# The MLP meta-learner is handled separately because it needs training first.

ENSEMBLE_REGISTRY = {
    "hard_voting":     hard_voting_batch,
    "soft_voting":     soft_voting_batch,
    "bayesian_fusion": sequential_bayesian_batch,
    "sugeno_fuzzy":    sugeno_fuzzy_batch,
}


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble fusion methods and compare their performance."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ensemble_config.yaml",
        help="Path to the ensemble config YAML file."
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively in addition to saving them."
    )
    parser.add_argument(
        "--train_mlp",
        action="store_true",
        help="Train and evaluate the MLP meta-learner. Requires all three "
             "models to have validation AND test prediction CSVs available."
    )
    return parser.parse_args()


# ==============================================================================
# CSV loading helpers
# ==============================================================================

def load_all_model_predictions(config: dict, split: str, logger) -> dict:
    """
    Load prediction CSV files for all models for a given split.

    Parameters
    ----------
    config : dict
        Ensemble config. Must have 'model_predictions' and 'dataset' sections.
    split : str
        Either "val" (predictions.csv) or "test" (test_predictions.csv).
        The val split is used to train the MLP; the test split is used
        to evaluate all methods.
    logger : logging.Logger

    Returns
    -------
    dict with keys matching model names in config["model_predictions"].
        Each value is a dict with keys: "image_names", "true_labels",
        "predictions", "probs" (as returned by load_predictions_csv).
    """
    class_names = config["dataset"]["class_names"]
    results_dir = config["results"]["dir"]

    # Map split name to the expected CSV filename.
    # "val" → "predictions.csv" (best validation epoch, saved during training)
    # "test" → "test_predictions.csv" (test set evaluation after training)
    filename_map = {
        "val":  "predictions.csv",
        "test": "test_predictions.csv",
    }
    csv_filename = filename_map[split]

    model_data = {}

    for model_name, path_str in config["model_predictions"].items():
        # Resolve "latest" keyword in the path to the actual timestamped folder.
        resolved = resolve_prediction_path(
            path_str  = path_str,
            base_dir  = results_dir.replace("ensemble", "training"),
            model_name = model_name
        )
        # Replace the filename part with the correct split filename.
        # resolved is e.g. "results/training/vgg16/2026-03-19_14-32/predictions.csv"
        # We replace just the filename, keeping the folder path.
        csv_path = resolved.parent / csv_filename

        if not csv_path.exists():
            logger.warning(
                f"  {model_name} {split} predictions not found: {csv_path}. "
                f"Skipping this model."
            )
            continue

        logger.info(f"  Loading {model_name} ({split}): {csv_path}")
        model_data[model_name] = load_predictions_csv(csv_path, class_names)

    return model_data


def validate_image_alignment(model_data: dict, logger) -> bool:
    """
    Check that all model CSVs contain predictions for the same images
    in the same order.

    This is the same assertion that existed in the original ensemble files
    but now produces a clear error message rather than a cryptic AssertionError.

    Parameters
    ----------
    model_data : dict
        Output of load_all_model_predictions().

    Returns
    -------
    bool
        True if all image lists match, False if there is a mismatch.
    """
    model_names  = list(model_data.keys())
    reference    = model_data[model_names[0]]["image_names"]

    for name in model_names[1:]:
        if model_data[name]["image_names"] != reference:
            logger.error(
                f"Image order mismatch between {model_names[0]} and {name}. "
                f"Make sure all models were evaluated on the same dataset "
                f"with shuffle=False."
            )
            return False

    logger.info(f"  Image alignment verified: {len(reference)} images, "
                f"{len(model_names)} models.")
    return True


# ==============================================================================
# Main evaluation function
# ==============================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ------------------------------------------------------------------
    args   = parse_args()
    config = load_config(args.config)

    class_names = config["dataset"]["class_names"]
    num_classes = config["dataset"]["num_classes"]

    # ------------------------------------------------------------------
    # 2. Create timestamped run directory
    # ------------------------------------------------------------------
    run_dir = make_run_dir(
        base_dir             = config["results"]["dir"],
        model_or_method_name = "comparison"
    )

    # ------------------------------------------------------------------
    # 3. Set up logging
    # ------------------------------------------------------------------
    logger = get_run_logger(run_dir, __name__, config=config)
    logger.info(f"Ensemble evaluation run directory: {run_dir}")
    logger.info(f"Config: {args.config}")

    # ------------------------------------------------------------------
    # 4. Load test set predictions for all models
    # ------------------------------------------------------------------
    # Test predictions are used to evaluate all ensemble methods.
    logger.info("Loading test set predictions...")
    test_data = load_all_model_predictions(config, split="test", logger=logger)

    if len(test_data) < 2:
        logger.error(
            "At least 2 models must have test predictions to run ensemble. "
            "Have you trained all models and run evaluate_final_model?"
        )
        return

    # Validate that all CSVs are aligned (same images, same order).
    if not validate_image_alignment(test_data, logger):
        return

    # Extract aligned arrays for ensemble input.
    # probs_arrays is a list of (N, num_classes) arrays, one per model.
    model_names  = list(test_data.keys())
    probs_arrays = [test_data[name]["probs"] for name in model_names]
    true_labels  = test_data[model_names[0]]["true_labels"]
    image_names  = test_data[model_names[0]]["image_names"]

    logger.info(f"Models loaded: {model_names}")
    logger.info(f"Test images:   {len(true_labels)}")

    # ------------------------------------------------------------------
    # 5. Run each ensemble method
    # ------------------------------------------------------------------
    methods_to_run = config["ensemble"]["methods"]
    logger.info(f"Running methods: {methods_to_run}")

    # Dictionaries to accumulate results across all methods.
    # method_metrics stores the full metrics dict per method.
    # cm_data stores true labels and predictions per method for plotting.
    method_metrics = {}
    cm_data        = {}

    for method_name in methods_to_run:

        if method_name == "mlp_meta_learner":
            # MLP is handled separately below — skip here.
            logger.info(f"  Skipping {method_name} (handled separately).")
            continue

        if method_name not in ENSEMBLE_REGISTRY:
            logger.warning(f"  Unknown method '{method_name}' — skipping.")
            continue

        logger.info(f"  Running: {method_name}")

        # Look up and call the batch fusion function.
        fusion_fn    = ENSEMBLE_REGISTRY[method_name]
        fused_probs  = fusion_fn(probs_arrays)

        # argmax along axis=1 gives the predicted class index per image.
        predictions  = np.argmax(fused_probs, axis=1).tolist()

        # Compute full metrics.
        metrics = calculate_metrics(predictions, true_labels, class_names)
        metrics["method"] = method_name

        method_metrics[method_name] = metrics
        cm_data[method_name] = {
            "true_labels": true_labels,
            "predictions": predictions,
        }

        # Save per-method metrics JSON.
        metrics_dir = run_dir / "metrics"
        save_json(metrics, metrics_dir / f"{method_name}.json")

        # Save per-method confusion matrix PNG.
        cm_dir = run_dir / "confusion_matrices"
        plot_confusion_matrix(
            true_labels = true_labels,
            predictions = predictions,
            class_names = class_names,
            title       = method_name.replace("_", " ").title(),
            save_dir    = cm_dir,
            filename    = f"{method_name}.png",
            show        = args.show_plots
        )

        logger.info(
            f"    Accuracy: {metrics['overall']['accuracy']:.4f} | "
            f"F1: {metrics['overall']['f1']:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. MLP meta-learner (optional, requires --train_mlp flag)
    # ------------------------------------------------------------------
    if args.train_mlp or "mlp_meta_learner" in methods_to_run:
        logger.info("Training MLP meta-learner...")

        # Load VALIDATION predictions to train the MLP.
        # These are different from the test predictions used above.
        val_data = load_all_model_predictions(config, split="val", logger=logger)

        if len(val_data) < 2:
            logger.warning(
                "Not enough validation prediction CSVs found. "
                "Skipping MLP meta-learner."
            )
        elif not validate_image_alignment(val_data, logger):
            logger.warning("Validation CSV alignment failed. Skipping MLP.")
        else:
            val_probs_arrays = [val_data[name]["probs"]       for name in model_names
                                if name in val_data]
            val_true_labels  = val_data[model_names[0]]["true_labels"]

            # Train MLP on validation predictions.
            mlp_save_path = run_dir / "mlp_weights.pth"
            mlp_model = train_mlp(
                probs_arrays  = val_probs_arrays,
                true_labels   = val_true_labels,
                num_classes   = num_classes,
                save_path     = mlp_save_path
            )

            # Evaluate MLP on TEST predictions (the same test data used
            # for all other methods — fair comparison).
            test_probs_for_mlp = [test_data[name]["probs"] for name in model_names
                                  if name in test_data]
            mlp_probs       = mlp_predict(mlp_model, test_probs_for_mlp)
            mlp_predictions = np.argmax(mlp_probs, axis=1).tolist()

            mlp_metrics = calculate_metrics(mlp_predictions, true_labels, class_names)
            mlp_metrics["method"] = "mlp_meta_learner"

            method_metrics["mlp_meta_learner"] = mlp_metrics
            cm_data["mlp_meta_learner"] = {
                "true_labels": true_labels,
                "predictions": mlp_predictions,
            }

            save_json(mlp_metrics, run_dir / "metrics" / "mlp_meta_learner.json")
            plot_confusion_matrix(
                true_labels = true_labels,
                predictions = mlp_predictions,
                class_names = class_names,
                title       = "MLP Meta-Learner",
                save_dir    = run_dir / "confusion_matrices",
                filename    = "mlp_meta_learner.png",
                show        = args.show_plots
            )

            logger.info(
                f"  MLP — Accuracy: {mlp_metrics['overall']['accuracy']:.4f} | "
                f"F1: {mlp_metrics['overall']['f1']:.4f}"
            )

    # ------------------------------------------------------------------
    # 7. Save comparison summary
    # ------------------------------------------------------------------
    if method_metrics:
        logger.info("Saving comparison summary...")

        # Build the flat comparison table (list of dicts, one per method).
        comparison_rows = build_comparison_table(method_metrics)

        # Save as JSON.
        save_json(
            {"methods": comparison_rows},
            run_dir / "metrics" / "comparison_summary.json"
        )

        # Save as CSV (convenient for quick inspection in Excel).
        save_metrics_csv(
            save_dir  = run_dir / "metrics",
            filename  = "comparison_summary.csv",
            metrics_list = comparison_rows
        )

    # ------------------------------------------------------------------
    # 8. Comparison plots
    # ------------------------------------------------------------------
    if len(method_metrics) > 1:
        logger.info("Generating comparison plots...")

        plots_dir = run_dir / "plots"

        # Side-by-side confusion matrices for all methods.
        plot_confusion_matrix_grid(
            all_metrics = cm_data,
            class_names = class_names,
            save_dir    = run_dir / "confusion_matrices",
            show        = args.show_plots
        )

        # Grouped bar chart: all metrics × all methods.
        plot_metric_comparison(
            comparison_rows = comparison_rows,
            save_dir        = plots_dir,
            show            = args.show_plots
        )

        # Per-class F1 comparison.
        plot_per_class_comparison(
            method_metrics = method_metrics,
            class_names    = class_names,
            save_dir       = plots_dir,
            show           = args.show_plots
        )

    # ------------------------------------------------------------------
    # 9. Final summary table printed to the log
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("ENSEMBLE EVALUATION COMPLETE")
    logger.info(f"{'Method':<25} {'Accuracy':>10} {'Precision':>10} "
                f"{'Recall':>10} {'F1':>10} {'F2':>10}")
    logger.info("-" * 70)
    for row in comparison_rows:
        logger.info(
            f"{row['Method']:<25} {row['Accuracy']:>10.4f} "
            f"{row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['F2']:>10.4f}"
        )
    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()
