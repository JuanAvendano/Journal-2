"""
pipeline/report_generator.py
------------------------------------------------------------------------------
Reads all test_metrics.json files produced by pipeline training runs and
generates a summary CSV plus diagnostic plots.

Called automatically at the end of run_pipeline.py, but can also be run
manually at any time:

    python pipeline/report_generator.py --results_dir results/training/

Output is written to:
    results/report_{timestamp}/
        summary_results.csv
        plot_A_learning_curves.png
        plot_A_model_comparison.png
        plot_B_binary_diagnostic.png
        plot_C_specialist_heatmap.png
        plot_C_model_comparison.png
        expert_assignment.txt

What the report generator expects in each test_metrics.json:
  - metrics["overall"]["accuracy"]         <- overall accuracy
  - metrics["overall"]["f1"]               <- macro F1
  - metrics["overall"]["f2"]               <- macro F2
  - metrics["overall"]["precision"]
  - metrics["overall"]["recall"]
  - metrics["overall"]["specificity"]
  - metrics["per_class"]["crack"]["f1"]    <- per-class F1 (and other metrics)
    metrics["per_class"]["efflorescence"]["f1"]
    ...
  - metrics["model"]                       <- model name string
  - metrics["experiment_id"]               <- e.g. "A1_VGG16"
  - metrics["group"]                       <- e.g. "A"

Only runs launched through the pipeline will have "experiment_id" and "group"
fields — manual runs are automatically skipped.
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for overnight runs
import matplotlib.pyplot as plt
import pandas as pd


# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("report")


# ==============================================================================
# Constants
# ==============================================================================

# Order that models appear in plots and tables
MODEL_ORDER = ["VGG16", "AlexNet", "ResNet50", "ensemble"]

# Consistent colours per model across all plots
MODEL_COLORS = {
    "vgg16":    "#4C72B0",
    "alexnet":  "#DD8452",
    "resnet50": "#55A868",
    "ensemble": "#C44E52",
}

# Damage class names in a stable order
CLASS_NAMES = ["crack", "efflorescence", "spalling", "undamaged"]


# ==============================================================================
# Result loading
# ==============================================================================

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Recursively scan results/training/ for test_metrics.json files produced
    by pipeline runs, and load them into a DataFrame.

    Only files that contain "experiment_id" and "group" keys are included —
    this filters out manual (non-pipeline) runs automatically.

    The metrics structure expected is:
        metrics["overall"]["accuracy"]
        metrics["per_class"]["crack"]["f1"]
        etc.

    Parameters
    ----------
    results_dir : Path to results/training/

    Returns
    -------
    pd.DataFrame  — one row per pipeline training run or ensemble evaluation.
    """
    records = []

    # Glob pattern: results/training/{model}/{experiment_id}/metrics/test_metrics.json
    for metrics_path in sorted(results_dir.rglob("metrics/test_metrics.json")):

        with open(metrics_path) as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {metrics_path}. Skipping.")
                continue

        # Skip manual (non-pipeline) runs — they lack these keys
        if "experiment_id" not in metrics or "group" not in metrics:
            continue

        experiment_id = metrics["experiment_id"]   # e.g. "A1_VGG16"
        group         = metrics["group"]            # e.g. "A"
        model_name    = metrics.get("model", "unknown")
        is_ensemble   = "ensemble" in experiment_id.lower()

        # Infer dataset_id from experiment_id:
        # "A1_VGG16"      -> "A1"
        # "C_crack_VGG16" -> "C_crack"
        # "A1_ensemble"   -> "A1"
        parts      = experiment_id.split("_")
        dataset_id = "_".join(parts[:-1])  # Everything except last token

        # --- Overall metrics --------------------------------------------------
        # Handle both nested structure (metrics["overall"]["accuracy"])
        # and flat structure (metrics["accuracy"]) for robustness.
        overall = metrics.get("overall", metrics)

        record = {
            "experiment_id": experiment_id,
            "group":         group,
            "dataset_id":    dataset_id,
            "model":         model_name,
            "is_ensemble":   is_ensemble,
            "accuracy":      overall.get("accuracy"),
            "precision":     overall.get("precision"),
            "recall":        overall.get("recall"),
            "specificity":   overall.get("specificity"),
            "f1":            overall.get("f1"),
            "f2":            overall.get("f2"),
        }

        # --- Per-class F1 scores ----------------------------------------------
        # Expected structure: metrics["per_class"]["crack"]["f1"]
        per_class = metrics.get("per_class", {})
        for class_name in CLASS_NAMES:
            class_data = per_class.get(class_name, {})
            record[f"f1_{class_name}"] = class_data.get("f1")

        records.append(record)

    if not records:
        logger.warning("No pipeline results found. Check that experiments have run.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Sort: group -> dataset_id -> individual models before ensemble
    df = df.sort_values(
        ["group", "dataset_id", "is_ensemble", "model"]
    ).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} run(s) across groups: "
                f"{sorted(df['group'].unique().tolist())}")
    return df


# ==============================================================================
# CSV export
# ==============================================================================

def save_summary_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the full results table as a CSV for easy inspection in Excel."""
    path = output_dir / "summary_results.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info(f"CSV saved: {path}")


# ==============================================================================
# Plot A — Learning curves
# ==============================================================================

def plot_learning_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Accuracy vs. dataset scale for Experiment A.

    Each line = one model. X-axis = dataset ID (A1 → A4).
    This is the core diagnostic plot: does accuracy improve and overfitting
    decrease as dataset size grows?
    """
    df_a = df[df["group"] == "A"].copy()
    if df_a.empty:
        logger.warning("No group A results — skipping learning curve plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in [m.lower() for m in MODEL_ORDER]:
        subset = df_a[df_a["model"] == model].sort_values("dataset_id")
        if subset.empty or subset["accuracy"].isna().all():
            continue
        ax.plot(
            subset["dataset_id"],
            subset["accuracy"],
            marker="o",
            label=model.upper(),
            color=MODEL_COLORS.get(model, "gray"),
            linewidth=2,
            markersize=7
        )

    ax.set_title("Experiment A — Accuracy vs. Dataset Size", fontsize=13)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "plot_A_learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ==============================================================================
# Plot B — Binary diagnostic
# ==============================================================================

def plot_binary_diagnostic(df: pd.DataFrame, output_dir: Path) -> None:
    """
    F1 score for VGG16 and AlexNet on the binary task (Experiment B).

    If F1 improves from B1 to B2: overfitting is data-driven (fixable).
    If F1 stays flat: overfitting is architectural (needs dropout / weight decay).
    This gives you a concrete, defensible conclusion for your paper.
    """
    df_b = df[df["group"] == "B"].copy()
    if df_b.empty:
        logger.warning("No group B results — skipping binary diagnostic plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for model in ["vgg16", "alexnet"]:
        subset = df_b[df_b["model"] == model].sort_values("dataset_id")
        if subset.empty:
            continue
        ax.plot(
            subset["dataset_id"],
            subset["f1"],
            marker="o",
            label=model.upper(),
            color=MODEL_COLORS.get(model, "gray"),
            linewidth=2,
            markersize=8
        )

    ax.set_title("Experiment B — Binary Diagnostic (Crack vs. Undamaged)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "plot_B_binary_diagnostic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ==============================================================================
# Plot C — Specialist heatmap
# ==============================================================================

def plot_specialist_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Per-class F1 heatmap for Experiment C specialist runs.

    One heatmap per model.
    Rows    = specialist config (C_crack, C_efflorescence, C_spalling)
    Columns = per-class F1 score on the test set

    This is the decision table for expert assignment:
    which model performs best on which damage type when specialised?
    """
    df_c = df[(df["group"] == "C") & (~df["is_ensemble"])].copy()
    if df_c.empty:
        logger.warning("No group C results — skipping specialist heatmap.")
        return

    models_present = [m for m in ["vgg16", "alexnet", "resnet50"]
                      if m in df_c["model"].values]

    if not models_present:
        return

    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(5 * len(models_present), 4),
        sharey=True
    )
    if len(models_present) == 1:
        axes = [axes]

    f1_cols   = [f"f1_{c}" for c in CLASS_NAMES if f"f1_{c}" in df_c.columns]
    col_labels = [c.replace("f1_", "") for c in f1_cols]

    for ax, model in zip(axes, models_present):
        subset = df_c[df_c["model"] == model].sort_values("dataset_id")
        row_labels = subset["dataset_id"].tolist()
        matrix = subset[f1_cols].values.astype(float)

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "black" if 0.3 < val < 0.8 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(model.upper(), fontsize=11)

    fig.colorbar(im, ax=axes[-1], label="F1 Score")
    fig.suptitle("Experiment C — Per-class F1 by Specialist Config", fontsize=13)
    fig.tight_layout()

    path = output_dir / "plot_C_specialist_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ==============================================================================
# Plot — Model comparison bar chart (one per group)
# ==============================================================================

def plot_model_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Side-by-side bar chart comparing F1 scores across all models and datasets
    within each experiment group. One figure per group.
    """
    for group in sorted(df["group"].unique()):
        df_g = df[df["group"] == group].copy()
        dataset_ids   = sorted(df_g["dataset_id"].unique())
        models_in_grp = [m for m in [m.lower() for m in MODEL_ORDER]
                         if m in df_g["model"].values]

        if not models_in_grp or not dataset_ids:
            continue

        x     = np.arange(len(dataset_ids))
        width = 0.8 / len(models_in_grp)

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(dataset_ids)), 5))

        for i, model in enumerate(models_in_grp):
            f1_vals = []
            for ds in dataset_ids:
                row = df_g[(df_g["dataset_id"] == ds) & (df_g["model"] == model)]
                f1_vals.append(
                    float(row["f1"].values[0]) if not row.empty and not row["f1"].isna().all() else 0
                )

            offset = (i - len(models_in_grp) / 2 + 0.5) * width
            bars   = ax.bar(
                x + offset, f1_vals, width,
                label=model.upper(),
                color=MODEL_COLORS.get(model, "gray"),
                edgecolor="white"
            )

            for bar, val in zip(bars, f1_vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7
                    )

        ax.set_title(f"Experiment {group} — F1 Score by Dataset and Model",
                     fontsize=13)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_ids)
        ax.set_ylim(0, 1.15)
        ax.legend(title="Model")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        path = output_dir / f"plot_{group}_model_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")


# ==============================================================================
# Expert assignment table (Experiment C)
# ==============================================================================

def compute_expert_assignment(df: pd.DataFrame, output_dir: Path) -> None:
    """
    For each damage class, find which model achieved the highest per-class F1
    when trained as the specialist for that class.

    Prints the result to console and saves it as a text file — this is the
    table you use to decide the expert assignment for the C2 ensemble.
    """
    df_c = df[(df["group"] == "C") & (~df["is_ensemble"])].copy()
    if df_c.empty:
        return

    lines = [
        "=" * 55,
        "EXPERT ASSIGNMENT TABLE  (Experiment C)",
        "Which model is best-suited for each damage type?",
        "=" * 55,
        ""
    ]

    damage_classes = ["crack", "efflorescence", "spalling"]

    for damage in damage_classes:
        col = f"f1_{damage}"
        if col not in df_c.columns:
            continue

        # Filter to rows where the specialist config targeted this damage
        # Dataset IDs for specialist configs: "C_crack", "C_efflorescence", "C_spalling"
        relevant = df_c[df_c["dataset_id"].str.lower().str.contains(damage[:4])]

        if relevant.empty or relevant[col].isna().all():
            lines.append(f"  {damage:20s}: no data available")
            continue

        best_idx = relevant[col].idxmax()
        best_row = relevant.loc[best_idx]

        line = (
            f"  {damage:20s}: {best_row['model'].upper():10s} "
            f"(F1 = {best_row[col]:.3f},  config = {best_row['dataset_id']})"
        )
        lines.append(line)

    lines += [
        "",
        "Use this assignment when building the C2 specialist ensemble:",
        "  -> Set class_weights in ensemble_eval.py to upweight each",
        "     model's specialist class during fusion.",
        "=" * 55,
    ]

    text = "\n".join(lines)
    logger.info("\n" + text)

    path = output_dir / "expert_assignment.txt"
    path.write_text(text, encoding="utf-8")
    logger.info(f"Saved: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate report from all pipeline experiment results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/training",
        help="Path to results/training/ directory."
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    # Create timestamped output folder inside results/
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = results_dir.parent / f"report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Report output: {report_dir}")

    # Load all pipeline results
    df = load_all_results(results_dir)
    if df.empty:
        logger.error("Nothing to report.")
        return

    # Generate all outputs
    save_summary_csv(df, report_dir)
    plot_learning_curves(df, report_dir)
    plot_binary_diagnostic(df, report_dir)
    plot_specialist_heatmap(df, report_dir)
    plot_model_comparison(df, report_dir)
    compute_expert_assignment(df, report_dir)

    logger.info(f"\nAll report files saved to: {report_dir}")


if __name__ == "__main__":
    main()
