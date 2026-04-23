"""
Created on Thursday Mar 19 2026
scripts/train.py
------------------------------------------------------------------------------
Entry point for training a single CNN model.

This script supports two usage modes:

  MODE 1 — Manual (original behaviour, unchanged):
    python scripts/train.py --model vgg16 --config configs/train_config.yaml

  MODE 2 — Pipeline (new):
    python scripts/train.py --pipeline_config pipeline/configs/experiment_A/A1_VGG16.yaml

    In pipeline mode, a single YAML file specifies everything: the model,
    dataset path, class weights, epochs, etc. This is what run_pipeline.py
    uses to drive all experiments automatically overnight.

    Key differences in pipeline mode:
      - The run directory is named after the experiment_id (e.g. "A1_VGG16")
        instead of a timestamp, so run_pipeline.py knows exactly where to
        find the output without having to guess the timestamp.
      - class_weights from the pipeline config are applied to CrossEntropyLoss,
        enabling the Experiment C specialist training.
      - experiment_id and group are stored in test_metrics.json so the
        report generator can organise results by experiment group.

Output structure:
  Manual mode  : results/training/vgg16/2026-03-19_14-32/
  Pipeline mode: results/training/vgg16/A1_VGG16/
    predictions/
        predictions.csv
        test_predictions.csv
    metrics/
        test_metrics.json
    curves/
        vgg16_loss_curve.png
        vgg16_accuracy_curve.png
    confusion_matrices/
        confusion_matrix.png
    run.log
"""

import argparse
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path so "from src.xxx" imports work.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils    import load_config, make_run_dir, save_json
from src.utils.logger      import get_run_logger
from src.data.dataloader   import get_dataloaders
from src.models.base_model import train_model, evaluate_final_model
from src.evaluation.metrics          import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.plots            import plot_training_curves


# ==============================================================================
# Model registry
# ==============================================================================

from src.models.vgg16    import load_vgg16
from src.models.resnet50 import load_resnet50
from src.models.alexnet  import load_alexnet

MODEL_REGISTRY = {
    "vgg16":    load_vgg16,
    "resnet50": load_resnet50,
    "alexnet":  load_alexnet,
}

MODEL_INPUT_SIZES = {
    "vgg16":    224,
    "resnet50": 224,
    "alexnet":  227,
}


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Both --model (manual mode) and --pipeline_config (pipeline mode) are
    optional here — we validate that exactly one is provided further below
    in main(), where we can print a clear error message.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN model for concrete damage classification."
    )

    # --- Manual mode arguments ------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default=None,                          # Not required — pipeline mode may omit it
        choices=list(MODEL_REGISTRY.keys()),
        help="(Manual mode) Model architecture to train: vgg16, resnet50, alexnet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="(Manual mode) Path to the base YAML config file."
    )

    # --- Pipeline mode argument -----------------------------------------------
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default=None,
        help="(Pipeline mode) Path to a pipeline experiment YAML config. "
             "When provided, --model and --config are ignored. "
             "The pipeline config must contain: experiment_id, group, model, "
             "dataset_path, test_set_path, num_classes, class_names, "
             "class_weights, epochs, batch_size, learning_rate, random_seed."
    )

    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively during training."
    )

    return parser.parse_args()


# ==============================================================================
# Pipeline config application
# ==============================================================================

def apply_pipeline_config(pipeline_cfg: dict, base_config: dict) -> dict:
    """
    Merge a flat pipeline YAML config into the nested base config dict.

    The pipeline YAML uses simple, flat keys (e.g. 'dataset_path', 'epochs').
    The base config uses nested keys (e.g. config['paths']['train_dir']).
    This function maps one to the other, so the rest of the code can keep
    reading from the same nested config structure as always.

    Parameters
    ----------
    pipeline_cfg : dict
        Contents of the pipeline YAML file (flat keys).
    base_config : dict
        Contents of configs/train_config.yaml (nested keys).
        This dict is modified in place and returned.

    Returns
    -------
    dict
        The updated base_config with pipeline values applied.
    """
    # --- Data paths -----------------------------------------------------------
    # train_config.yaml uses three separate path keys:
    #   paths.train  — folder with class subfolders for training
    #   paths.val    — folder with class subfolders for validation
    #   paths.test   — folder with class subfolders for testing
    #
    # The pipeline YAML stores these as train_path, val_path, test_set_path.
    # They are derived from dataset_path in generate_configs.py by appending
    # the subfolder names (default: "train/" and "val/").
    if "train_path" in pipeline_cfg:
        base_config["paths"]["train"] = pipeline_cfg["train_path"]

    if "val_path" in pipeline_cfg:
        base_config["paths"]["val"] = pipeline_cfg["val_path"]

    if "test_set_path" in pipeline_cfg:
        base_config["paths"]["test"] = pipeline_cfg["test_set_path"]

    # --- Dataset properties ---------------------------------------------------
    if "num_classes" in pipeline_cfg:
        base_config["dataset"]["num_classes"] = pipeline_cfg["num_classes"]

    if "class_names" in pipeline_cfg:
        base_config["dataset"]["class_names"] = pipeline_cfg["class_names"]

    # --- Training hyperparameters ---------------------------------------------
    if "epochs" in pipeline_cfg:
        base_config["training"]["epochs"] = pipeline_cfg["epochs"]

    if "batch_size" in pipeline_cfg:
        base_config["training"]["batch_size"] = pipeline_cfg["batch_size"]

    if "learning_rate" in pipeline_cfg:
        base_config["training"]["learning_rate"] = pipeline_cfg["learning_rate"]

    if "random_seed" in pipeline_cfg:
        # The base config may call this 'seed' or 'random_seed' — adjust if needed.
        base_config["training"]["random_seed"] = pipeline_cfg["random_seed"]

    # --- Class weights --------------------------------------------------------
    # This key does NOT exist in the original base config — we add it here.
    # It is read later when building CrossEntropyLoss.
    # For Experiments A and B all weights are 1.0 (standard loss).
    # For Experiment C the specialist class gets weight 3.0.
    base_config["training"]["class_weights"] = pipeline_cfg.get(
        "class_weights", [1.0] * pipeline_cfg.get("num_classes", 4)
    )

    return base_config


# ==============================================================================
# Loss function builder
# ==============================================================================

def build_criterion(config: dict, device: torch.device) -> nn.Module:
    """
    Build the CrossEntropyLoss criterion, applying class weights if specified.

    When class_weights are all 1.0 (Experiments A and B) this is identical
    to plain CrossEntropyLoss() — no behavioural change.

    When class_weights differ (Experiment C), errors on the high-weight class
    cost proportionally more, nudging the model to specialise.

    Parameters
    ----------
    config : dict
        The (merged) training config. Reads config['training']['class_weights'].
    device : torch.device
        The weights tensor must live on the same device as the model.

    Returns
    -------
    nn.CrossEntropyLoss
    """
    weights = config["training"].get("class_weights", None)

    if weights is not None and any(w != 1.0 for w in weights):
        # Convert list of floats to a CUDA/CPU tensor
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        # Standard unweighted loss — same as before
        criterion = nn.CrossEntropyLoss()

    return criterion


# ==============================================================================
# Main training function
# ==============================================================================

def main():

    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    args = parse_args()

    # Validate: exactly one of --model or --pipeline_config must be given
    if args.pipeline_config is None and args.model is None:
        raise ValueError(
            "You must provide either --model (manual mode) or "
            "--pipeline_config (pipeline mode)."
        )

    # ------------------------------------------------------------------
    # 2. Determine mode and load configs
    # ------------------------------------------------------------------
    pipeline_mode = args.pipeline_config is not None

    # Always load the base config — it contains paths, augmentation settings,
    # and other fields the pipeline YAML does not override.
    base_config_path = "configs/train_config.yaml"
    config = load_config(base_config_path)

    if pipeline_mode:
        # Read the flat pipeline YAML
        with open(args.pipeline_config) as f:
            pipeline_cfg = yaml.safe_load(f)

        # Merge pipeline values into the base config
        config = apply_pipeline_config(pipeline_cfg, config)

        # Extract pipeline-specific metadata
        experiment_id = pipeline_cfg["experiment_id"]   # e.g. "A1_VGG16"
        group         = pipeline_cfg["group"]            # e.g. "A"
        model_name    = pipeline_cfg["model"]            # e.g. "vgg16"

    else:
        # Manual mode — original behaviour
        model_name    = args.model
        experiment_id = None
        group         = None

    # Apply model name and input size to config (same as before)
    config["model"]["name"]       = model_name
    config["model"]["input_size"] = MODEL_INPUT_SIZES[model_name]

    # ------------------------------------------------------------------
    # 3. Create the run directory
    # ------------------------------------------------------------------
    # In pipeline mode we use the experiment_id as the folder name instead
    # of a timestamp. This makes the output location predictable, so
    # run_pipeline.py and report_generator.py can find results without
    # having to scan for the newest folder.
    #
    # Pipeline result: results/training/vgg16/A1_VGG16/
    # Manual result:   results/training/vgg16/2026-03-19_14-32/   (unchanged)

    if pipeline_mode:
        # config["paths"]["results"] is "results/training/" in train_config.yaml,
        # so we must NOT add another "training/" level here.
        # Final path: results/training/vgg16/A1_VGG16/
        run_dir = (
            Path(config["paths"]["results"])
            / model_name
            / experiment_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Original behaviour — timestamped folder
        run_dir = make_run_dir(
            base_dir             = config["paths"]["results"],
            model_or_method_name = model_name
        )

    # ------------------------------------------------------------------
    # 4. Set up logging
    # ------------------------------------------------------------------
    logger = get_run_logger(run_dir, __name__,config=config)
    logger.info(f"Run directory  : {run_dir}")
    logger.info(f"Model          : {model_name}")
    logger.info(f"Pipeline mode  : {pipeline_mode}")
    if pipeline_mode:
        logger.info(f"Experiment ID  : {experiment_id}")
        logger.info(f"Group          : {group}")
        logger.info(f"Class weights  : {config['training']['class_weights']}")

    # ------------------------------------------------------------------
    # 5. Detect device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 6. Load data
    # ------------------------------------------------------------------
    logger.info("Loading datasets...")
    loaders     = get_dataloaders(config)
    class_names = config["dataset"]["class_names"]
    num_classes = config["dataset"]["num_classes"]

    # ------------------------------------------------------------------
    # 7. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {model_name}")
    load_fn = MODEL_REGISTRY[model_name]
    model   = load_fn(num_classes=num_classes, device=device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.1f}%)")

    # ------------------------------------------------------------------
    # 8. Define loss function and optimiser
    # ------------------------------------------------------------------
    # build_criterion() returns plain CrossEntropyLoss when all weights
    # are 1.0 (Experiments A and B), and weighted loss for Experiment C.
    criterion = build_criterion(config, device)
    logger.info(f"Loss function  : {criterion}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # ------------------------------------------------------------------
    # 9. Train the model
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    history = train_model(
        model      = model,
        model_name = model_name,
        loaders    = loaders,
        criterion  = criterion,
        optimizer  = optimizer,
        config     = config,
        run_dir    = run_dir
    )

    # ------------------------------------------------------------------
    # 10. Plot training curves
    # ------------------------------------------------------------------
    logger.info("Saving training curves...")
    plot_training_curves(
        history    = history,
        model_name = model_name,
        save_dir   = run_dir / "curves",
        show       = args.show_plots
    )

    # ------------------------------------------------------------------
    # 11. Evaluate on the test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set...")
    predictions, true_labels, all_probs, filenames = evaluate_final_model(
        model       = model,
        model_name  = model_name,
        test_loader = loaders["test"],
        criterion   = criterion,
        config      = config,
        run_dir     = run_dir
    )

    # ------------------------------------------------------------------
    # 12. Compute and save metrics
    # ------------------------------------------------------------------
    logger.info("Computing test set metrics...")
    metrics = calculate_metrics(predictions, true_labels, class_names)

    # Store run metadata — this makes the JSON self-contained and allows
    # report_generator.py to organise results by experiment group without
    # needing to parse folder names.
    metrics["model"]      = model_name
    metrics["run_dir"]    = str(run_dir)
    metrics["config"]     = base_config_path

    # Pipeline-specific metadata (None in manual mode — JSON skips None values)
    if pipeline_mode:
        metrics["experiment_id"]   = experiment_id
        metrics["group"]           = group
        metrics["class_weights"]   = config["training"]["class_weights"]
        metrics["pipeline_config"] = str(args.pipeline_config)

    metrics_dir = run_dir / "metrics"
    save_json(metrics, metrics_dir / "test_metrics.json")

    # ------------------------------------------------------------------
    # 13. Confusion matrix
    # ------------------------------------------------------------------
    logger.info("Saving confusion matrix...")
    plot_confusion_matrix(
        true_labels = true_labels,
        predictions = predictions,
        class_names = class_names,
        title       = model_name.upper(),
        save_dir    = run_dir / "confusion_matrices",
        show        = args.show_plots
    )

    # ------------------------------------------------------------------
    # 14. Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Model      : {model_name}")
    if pipeline_mode:
        logger.info(f"  Experiment : {experiment_id} (Group {group})")
    logger.info(f"  Accuracy   : {metrics['overall']['accuracy']:.4f}")
    logger.info(f"  F1 Score   : {metrics['overall']['f1']:.4f}")
    logger.info(f"  F2 Score   : {metrics['overall']['f2']:.4f}")
    logger.info(f"  Run folder : {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
