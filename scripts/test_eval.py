"""
scripts/test_eval.py
------------------------------------------------------------------------------
Test-only evaluation script — NO training.

Use this script when you already have a trained .pth checkpoint and want to
run it against a (possibly new) test set to get metrics, a confusion matrix,
and a predictions CSV.

This is useful when you:
  - Have retrained or extended your dataset and want to re-evaluate old weights
    on a new test split without retraining.
  - Want to compare the same checkpoint against multiple different test sets.
  - Want to evaluate quickly without touching train.py at all.

Usage:
    python scripts/test_eval.py \
        --model   vgg16 \
        --weights saved_models/vgg16/best.pth \
        --config  configs/train_config.yaml

    # Override the test folder from the config with a different path:
    python scripts/test_eval.py --model    resnet50 --weights  saved_models/resnet50/best.pth --config   configs/train_config.yaml --test_dir D:/JCA/07-Data/01_Concrete/Test_set_dacl/

    # Evaluate all three models in sequence (run once per model):
    python scripts/test_eval.py --model vgg16    --weights saved_models/vgg16/best.pth    --config configs/train_config.yaml --test_dir D:/JCA/07-Data/01_Concrete/Test_set_dacl/
    python scripts/test_eval.py --model resnet50 --weights saved_models/resnet50/best.pth --config configs/train_config.yaml --test_dir D:/JCA/07-Data/01_Concrete/Test_set_dacl/
    python scripts/test_eval.py --model alexnet  --weights saved_models/alexnet/best.pth  --config configs/train_config.yaml --test_dir D:/JCA/07-Data/01_Concrete/Test_set_dacl/

Arguments:
    --model    : Architecture to rebuild. One of: vgg16, resnet50, alexnet.
    --weights  : Path to the .pth checkpoint file to load.
    --config   : Path to the YAML config file (same one used during training).
    --test_dir : (Optional) Override the test path in the config YAML.
                 Useful when you want to evaluate on a new test set without
                 editing the config file.
    --show_plots : Display plots interactively (default: save to disk only).

Output structure:
    results/test_eval/vgg16/2026-04-27_14-32/
        predictions/
            test_predictions.csv
        metrics/
            test_metrics.json
        confusion_matrices/
            confusion_matrix.png
        run.log
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so "from src.xxx" imports work when
# the script is run from the repo root with "python scripts/evaluate.py".
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils    import load_config, make_run_dir, save_json
from src.utils.logger      import get_run_logger
from src.data.dataloader   import get_dataloaders
from src.models.base_model import load_checkpoint, evaluate_final_model
from src.evaluation.metrics          import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix


# ==============================================================================
# Model registry  (identical to train.py — add new architectures here)
# ==============================================================================

from src.models.vgg16    import load_vgg16
from src.models.resnet50 import load_resnet50
from src.models.alexnet  import load_alexnet

MODEL_REGISTRY = {
    "vgg16":    load_vgg16,
    "resnet50": load_resnet50,
    "alexnet":  load_alexnet,
}

# Each architecture expects a specific input resolution.
# AlexNet: 227×227.  VGG16 and ResNet50: 224×224.
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
    Parse command-line arguments for the evaluation script.

    Returns
    -------
    argparse.Namespace
        Attributes: model, weights, config, test_dir, show_plots.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN on a test set (no training)."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture to rebuild. One of: vgg16, resnet50, alexnet."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file to load (e.g. saved_models/vgg16/best.pth)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the YAML config file used during training."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help=(
            "Override the test folder path from the config. "
            "Use this to evaluate on a new test set without editing the YAML. "
            "Example: --test_dir D:/JCA/.../B/03-test/"
        )
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively (default: save to disk only)."
    )

    return parser.parse_args()


# ==============================================================================
# Main evaluation function
# ==============================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ------------------------------------------------------------------
    args       = parse_args()
    config     = load_config(args.config)
    model_name = args.model
    weights_path = Path(args.weights)

    # Validate that the checkpoint file actually exists before doing anything.
    # A clear error here is much better than a cryptic crash later.
    if not weights_path.exists():
        print(f"[ERROR] Checkpoint not found: {weights_path}")
        print("        Double-check the --weights path and try again.")
        sys.exit(1)

    # Override the model name and input size from the CLI argument.
    config["model"]["name"]       = model_name
    config["model"]["input_size"] = MODEL_INPUT_SIZES[model_name]

    # ------------------------------------------------------------------
    # 2. Optionally override the test directory
    # ------------------------------------------------------------------
    # If --test_dir is provided, we replace the test path in the config.
    # This means the dataloader will load from the new test set, but all
    # other settings (normalisation, batch size, class names) stay the same.
    if args.test_dir is not None:
        config["paths"]["test"] = args.test_dir

    # ------------------------------------------------------------------
    # 3. Create a timestamped output directory under results/evaluation/
    # ------------------------------------------------------------------
    # We write to a separate "evaluation" subfolder so that these runs
    # never overwrite or mix with the original training run outputs.
    eval_results_base = "results/test_eval/"
    run_dir = make_run_dir(
        base_dir             = eval_results_base,
        model_or_method_name = model_name
    )

    # ------------------------------------------------------------------
    # 4. Set up logging
    # ------------------------------------------------------------------
    logger = get_run_logger(run_dir, __name__, config=config)
    logger.info("=" * 60)
    logger.info("EVALUATION-ONLY RUN  (no training)")
    logger.info("=" * 60)
    logger.info(f"Model:        {model_name}")
    logger.info(f"Weights:      {weights_path}")
    logger.info(f"Config:       {args.config}")
    logger.info(f"Test folder:  {config['paths']['test']}")
    logger.info(f"Output dir:   {run_dir}")

    # ------------------------------------------------------------------
    # 5. Detect device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 6. Build only the test dataloader
    # ------------------------------------------------------------------
    # get_dataloaders() normally builds train / val / test loaders.
    # We still call it the usual way — the train and val loaders are
    # built but simply never used. This avoids having to modify dataloader.py.
    # The overhead is negligible since we are not iterating over them.
    logger.info("Building test dataloader...")
    loaders     = get_dataloaders(config)
    class_names = config["dataset"]["class_names"]
    num_classes = config["dataset"]["num_classes"]

    # ------------------------------------------------------------------
    # 7. Rebuild the model architecture
    # ------------------------------------------------------------------
    # We must rebuild the same architecture that was used during training
    # before we can load the saved weights into it.
    logger.info(f"Rebuilding architecture: {model_name}")
    load_fn = MODEL_REGISTRY[model_name]
    model   = load_fn(num_classes=num_classes, device=device)

    # ------------------------------------------------------------------
    # 8. Load the saved weights directly from the path you specified
    # ------------------------------------------------------------------
    # We use load_checkpoint() from base_model.py, but we pass the weights
    # path directly instead of constructing it from saved_models/<model>/.
    # This is the key difference from evaluate_final_model() — you are in
    # full control of which checkpoint file is used.
    logger.info(f"Loading weights from: {weights_path}")
    model = load_checkpoint(model, weights_path, device)
    model.to(device)
    model.eval()   # Switch off dropout and batch norm training mode

    # ------------------------------------------------------------------
    # 9. Run inference on the test set
    # ------------------------------------------------------------------
    # We replicate the core of evaluate_final_model() here directly, rather
    # than calling evaluate_final_model() itself, because that function
    # internally re-loads the checkpoint from saved_models/<model>/best.pth.
    # Since we have already loaded the weights above (step 8), we want to
    # avoid that and use the model as-is.
    import os
    import numpy as np

    criterion  = nn.CrossEntropyLoss()   # Needed only to compute test loss

    running_loss = 0.0
    correct      = 0
    total        = 0
    predictions  = []
    labels_list  = []
    all_probs    = []
    filenames    = []

    logger.info("Running inference on test set...")

    with torch.no_grad():
        # torch.no_grad() disables gradient computation — we are doing
        # inference only, so we do not need to build the computational graph.
        for inputs, labels, paths in loaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: get raw logits from the model.
            outputs = model(inputs)

            # Collect true labels and filenames.
            labels_list += labels.tolist()
            filenames   += [os.path.basename(p) for p in paths]

            # Convert logits to class probabilities via softmax.
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            # Compute test loss (for logging only — not saved to metrics).
            loss          = criterion(outputs, labels)
            running_loss += loss.item()

            # Get the predicted class index (the class with the highest logit).
            _, predicted = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy().tolist())

    test_loss = running_loss / len(loaders["test"])
    test_acc  = correct / total
    logger.info(f"Test Loss:     {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Concatenate per-batch probability arrays into one (N, num_classes) array.
    all_probs_array = np.concatenate(all_probs, axis=0)

    # ------------------------------------------------------------------
    # 10. Save predictions CSV
    # ------------------------------------------------------------------
    from src.utils.io_utils import save_predictions_csv

    preds_dir = run_dir / "predictions"
    save_predictions_csv(
        save_dir         = preds_dir,
        filename         = "test_predictions.csv",
        image_names      = filenames,
        true_labels      = labels_list,
        predicted_labels = predictions,
        class_probs      = all_probs_array,
        class_names      = class_names
    )
    logger.info(f"Predictions saved to: {preds_dir / 'test_predictions.csv'}")

    # ------------------------------------------------------------------
    # 11. Compute and save metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")
    metrics = calculate_metrics(predictions, labels_list, class_names)

    # Add metadata so the JSON is self-contained.
    metrics["model"]       = model_name
    metrics["weights"]     = str(weights_path)
    metrics["test_folder"] = config["paths"]["test"]
    metrics["run_dir"]     = str(run_dir)

    metrics_dir = run_dir / "metrics"
    save_json(metrics, metrics_dir / "test_metrics.json")
    logger.info(f"Metrics saved to: {metrics_dir / 'test_metrics.json'}")

    # ------------------------------------------------------------------
    # 12. Plot and save confusion matrix
    # ------------------------------------------------------------------
    logger.info("Saving confusion matrix...")
    cm_dir = run_dir / "confusion_matrices"
    plot_confusion_matrix(
        true_labels = labels_list,
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
    logger.info("EVALUATION COMPLETE")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Weights:    {weights_path}")
    logger.info(f"  Accuracy:   {metrics['overall']['accuracy']:.4f}")
    logger.info(f"  F1 Score:   {metrics['overall']['f1']:.4f}")
    logger.info(f"  F2 Score:   {metrics['overall']['f2']:.4f}")
    logger.info(f"  Output dir: {run_dir}")
    logger.info("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()
