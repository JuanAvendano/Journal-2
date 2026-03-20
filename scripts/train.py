"""
Created on Thursday Mar 19 2026
scripts/train.py
------------------------------------------------------------------------------
Entry point for training a single CNN model.

This script ties together everything built in src/ into one runnable file.
It reads the configuration, builds the data loaders, loads the model,
runs the training loop, evaluates on the test set, and saves all outputs
(weights, predictions, metrics, curves, confusion matrix) to a timestamped
run directory.

Usage:
    python scripts/train.py --model vgg16 --config configs/train_config.yaml

    # To train all three models sequentially (run this three times):
    python scripts/train.py --model vgg16    --config configs/train_config.yaml
    python scripts/train.py --model resnet50 --config configs/train_config.yaml
    python scripts/train.py --model alexnet  --config configs/train_config.yaml

Arguments:
    --model   : Which architecture to train. One of: vgg16, resnet50, alexnet.
                This overrides the 'model.name' field in the config file,
                so you only need one config file for all three models.
    --config  : Path to the YAML config file. Defaults to
                configs/train_config.yaml if not specified.
    --show_plots : Add this flag to display plots interactively during
                training (useful for debugging). By default plots are only
                saved to disk.

Output structure (one timestamped folder per run):
    results/training/vgg16/2026-03-19_14-32/
        predictions/
            predictions.csv       ← best validation predictions (for ensemble)
            test_predictions.csv  ← test set predictions
        metrics/
            test_metrics.json     ← full metrics dict from calculate_metrics()
        curves/
            vgg16_loss_curve.png
            vgg16_accuracy_curve.png
        confusion_matrices/
            confusion_matrix.png
        run.log                   ← full log of this training run
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so that "from src.xxx" imports work
# when the script is run from the repo root with "python scripts/train.py".
# ---------------------------------------------------------------------------
# Path(__file__) is the path to this file (scripts/train.py).
# .resolve() makes it absolute. .parent is the scripts/ folder.
# .parent again is the repo root. We insert it at position 0 in sys.path
# so it takes priority over any other installed packages with the same name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils   import load_config, make_run_dir, save_json
from src.utils.logger     import get_run_logger
from src.data.dataloader  import get_dataloaders
from src.models.base_model import train_model, evaluate_final_model
from src.evaluation.metrics         import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.plots            import plot_training_curves


# ==============================================================================
# Model registry
# ==============================================================================
# A dictionary that maps model name strings to their loader functions.
# This pattern is called a "registry" — it lets us select the right function
# at runtime based on the --model argument without needing a chain of
# if/elif statements.
#
# To add a new model (e.g. VGG19) in the future:
#   1. Create src/models/vgg19.py with a load_vgg19() function.
#   2. Import it here and add "vgg19": load_vgg19 to this dict.
#   3. No other changes needed anywhere.

from src.models.vgg16    import load_vgg16
from src.models.resnet50 import load_resnet50
from src.models.alexnet  import load_alexnet

MODEL_REGISTRY = {
    "vgg16":    load_vgg16,
    "resnet50": load_resnet50,
    "alexnet":  load_alexnet,
}

# Input sizes required by each architecture.
# AlexNet needs 227x227; VGG16 and ResNet50 need 224x224.
# This overrides whatever input_size is in the config when the model name
# is passed via --model, ensuring the correct size is always used.
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

    argparse is Python's standard library for handling command-line arguments.
    It automatically generates a --help message and validates argument types.

    Returns
    -------
    argparse.Namespace
        Object with attributes: args.model, args.config, args.show_plots.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN model for concrete damage classification."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),   # Only allow valid model names
        help="Model architecture to train. One of: vgg16, resnet50, alexnet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config YAML file."
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",   # store_true means: if flag is present → True
        help="Display plots interactively during training."
    )

    return parser.parse_args()


# ==============================================================================
# Main training function
# ==============================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ------------------------------------------------------------------
    args   = parse_args()
    config = load_config(args.config)

    # Override the model name in the config with the CLI argument.
    # This means you only need one config file — the --model flag selects
    # the architecture, and the config provides all hyperparameters.
    model_name = args.model
    config["model"]["name"]       = model_name
    config["model"]["input_size"] = MODEL_INPUT_SIZES[model_name]

    # ------------------------------------------------------------------
    # 2. Create the timestamped run directory
    # ------------------------------------------------------------------
    # All outputs for this run (logs, metrics, curves, predictions) go
    # into this directory. make_run_dir() creates it and returns the path.
    run_dir = make_run_dir(
        base_dir             = config["paths"]["results"],
        model_or_method_name = model_name
    )

    # ------------------------------------------------------------------
    # 3. Set up logging
    # ------------------------------------------------------------------
    # get_run_logger() creates a logger that writes to both the terminal
    # and a run.log file inside the run directory.
    logger = get_run_logger(run_dir, __name__)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Model:         {model_name}")
    logger.info(f"Config:        {args.config}")

    # ------------------------------------------------------------------
    # 4. Detect device (GPU or CPU)
    # ------------------------------------------------------------------
    # torch.cuda.is_available() returns True if a CUDA-capable GPU is
    # detected and CUDA drivers are installed.
    # On your machine with the GTX 1060 this should return True.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 5. Load data
    # ------------------------------------------------------------------
    logger.info("Loading datasets...")
    loaders     = get_dataloaders(config)
    class_names = config["dataset"]["class_names"]
    num_classes = config["dataset"]["num_classes"]

    # ------------------------------------------------------------------
    # 6. Load model
    # ------------------------------------------------------------------
    # Look up the loader function from the registry and call it.
    # MODEL_REGISTRY["vgg16"] is load_vgg16, so this is equivalent to:
    #   model = load_vgg16(num_classes=4, device=device)
    logger.info(f"Loading model: {model_name}")
    load_fn = MODEL_REGISTRY[model_name]
    model   = load_fn(num_classes=num_classes, device=device)

    # Log how many parameters are being trained vs frozen.
    # This is useful to verify that the backbone is indeed frozen.
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.1f}%)")

    # ------------------------------------------------------------------
    # 7. Define loss function and optimiser
    # ------------------------------------------------------------------
    # CrossEntropyLoss is the standard loss for multi-class classification.
    # It combines a softmax activation with a negative log-likelihood loss,
    # so we do NOT apply softmax to the model output before passing it here.
    criterion = nn.CrossEntropyLoss()

    # Adam (Adaptive Moment Estimation) is the standard optimiser for
    # fine-tuning pretrained CNNs. It adapts the learning rate per parameter
    # and is generally more robust than plain SGD for transfer learning.
    # model.parameters() returns only the parameters that have
    # requires_grad=True — i.e. only the unfrozen layers.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    # ------------------------------------------------------------------
    # 8. Train the model
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
    # 9. Plot and save training curves
    # ------------------------------------------------------------------
    logger.info("Saving training curves...")
    curves_dir = run_dir / "curves"
    plot_training_curves(
        history    = history,
        model_name = model_name,
        save_dir   = curves_dir,
        show       = args.show_plots
    )

    # ------------------------------------------------------------------
    # 10. Evaluate on the test set
    # ------------------------------------------------------------------
    # evaluate_final_model() loads best.pth internally, so we are always
    # evaluating the best checkpoint, not the last one.
    logger.info("Evaluating on test set...")
    predictions, true_labels, all_probs, filenames = evaluate_final_model(
        model      = model,
        model_name = model_name,
        test_loader = loaders["test"],
        criterion  = criterion,
        config     = config,
        run_dir    = run_dir
    )

    # ------------------------------------------------------------------
    # 11. Compute and save metrics
    # ------------------------------------------------------------------
    logger.info("Computing test set metrics...")
    metrics = calculate_metrics(predictions, true_labels, class_names)

    # Add run metadata to the metrics dict before saving.
    # This means the JSON file is self-contained — you can open it months
    # later and know exactly which model and config produced these numbers.
    metrics["model"]      = model_name
    metrics["run_dir"]    = str(run_dir)
    metrics["config"]     = args.config

    metrics_dir = run_dir / "metrics"
    save_json(metrics, metrics_dir / "test_metrics.json")

    # ------------------------------------------------------------------
    # 12. Plot and save confusion matrix
    # ------------------------------------------------------------------
    logger.info("Saving confusion matrix...")
    cm_dir = run_dir / "confusion_matrices"
    plot_confusion_matrix(
        true_labels = true_labels,
        predictions = predictions,
        class_names = class_names,
        title       = model_name.upper(),
        save_dir    = cm_dir,
        show        = args.show_plots
    )

    # ------------------------------------------------------------------
    # 13. Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Accuracy:   {metrics['overall']['accuracy']:.4f}")
    logger.info(f"  F1 Score:   {metrics['overall']['f1']:.4f}")
    logger.info(f"  F2 Score:   {metrics['overall']['f2']:.4f}")
    logger.info(f"  Run folder: {run_dir}")
    logger.info("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================
# This block only runs when the script is executed directly:
#   python scripts/train.py --model vgg16
# It does NOT run if this file is imported by another module.
# This is standard Python practice for scripts.

if __name__ == "__main__":
    main()
