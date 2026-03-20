"""
scripts/deploy.py
------------------------------------------------------------------------------
Entry point for deployment — running the ensemble on new, unlabelled bridge
inspection images to flag which ones contain damage.

This script is the interface between this repository and the broader inspection
pipeline. It takes a folder of raw images, runs them through all three trained
CNN models, applies the configured fusion method, and writes a JSON file
listing which images are flagged as damaged and with what confidence.

That JSON file is then consumed by the segmentation repository, which generates
pixel-level masks for the flagged images.

Usage (standalone):
    python scripts/deploy.py --input path/to/images
                             --config configs/deploy_config.yaml

Usage (via pipeline.yaml):
    The pipeline.yaml at the repo root calls this script automatically as
    part of the full inspection workflow.

Arguments:
    --input       : Path to a folder containing the images to inspect.
                    Images can be .jpg, .jpeg, or .png.
    --config      : Path to deploy_config.yaml. Defaults to the standard location.
    --output      : Optional override for the output JSON path. If not given,
                    the path from deploy_config.yaml is used.

Output JSON structure:
    {
        "run_info": {
            "input_dir":      "path/to/images",
            "fusion_method":  "soft_voting",
            "threshold":      0.7,
            "total_images":   150,
            "flagged_count":  42,
            "timestamp":      "2026-03-19_15-00"
        },
        "flagged_images": [
            {
                "filename":    "img_001.jpg",
                "prediction":  "crack",
                "confidence":  0.91,
                "all_probs": {
                    "crack":         0.91,
                    "efflorescence": 0.04,
                    "spalling":      0.03,
                    "undamaged":     0.02
                }
            },
            ...
        ],
        "all_predictions": [
            {
                "filename":   "img_001.jpg",
                "prediction": "crack",
                "confidence": 0.91,
                "flagged":    true
            },
            ...
        ]
    }

The "flagged_images" list contains only images predicted as damaged above the
confidence threshold. The "all_predictions" list contains every image, which
allows the downstream pipeline to audit the results or adjust the threshold.
"""

import argparse
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for "from src.xxx" imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils  import load_config, save_json
from src.utils.logger    import get_run_logger

from src.models.base_model import load_checkpoint
from src.models.vgg16      import load_vgg16
from src.models.resnet50   import load_resnet50
from src.models.alexnet    import load_alexnet

from src.ensemble.hard_voting     import hard_voting_batch
from src.ensemble.soft_voting     import soft_voting_batch
from src.ensemble.bayesian_fusion import sequential_bayesian_batch

from src.data.augmentations import get_eval_transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# ==============================================================================
# Registries
# ==============================================================================

MODEL_LOADERS = {
    "vgg16":    load_vgg16,
    "resnet50": load_resnet50,
    "alexnet":  load_alexnet,
}

ENSEMBLE_REGISTRY = {
    "hard_voting":     hard_voting_batch,
    "soft_voting":     soft_voting_batch,
    "bayesian_fusion": sequential_bayesian_batch,
}

# Supported image file extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble inference on unlabelled bridge images."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the folder containing images to inspect."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deploy_config.yaml",
        help="Path to the deploy config YAML file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for the output JSON file path."
    )
    return parser.parse_args()


# ==============================================================================
# Dataset for unlabelled images
# ==============================================================================

class UnlabelledImageDataset(torch.utils.data.Dataset):
    """
    A simple PyTorch Dataset for loading unlabelled images from a flat folder.

    Unlike ImageFolder (which requires class subfolders), this dataset reads
    all images directly from a single directory. There are no labels because
    we are doing inference on new, unseen images.

    It returns (image_tensor, filename) pairs so we can track which prediction
    belongs to which image file.
    """

    def __init__(self, image_dir: Path, transform):
        """
        Parameters
        ----------
        image_dir : Path
            Directory containing image files (flat, no subfolders needed).
        transform : torchvision.transforms.Compose
            Preprocessing pipeline (resize + normalise).
        """
        self.transform = transform

        # Collect all image files in the directory.
        # We sort them so the order is deterministic across runs.
        self.image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix in IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in '{image_dir}'. "
                f"Supported formats: {IMAGE_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Load and transform one image.

        Returns
        -------
        tuple: (image_tensor, filename)
            image_tensor : torch.Tensor, shape (C, H, W)
            filename     : str, just the file basename e.g. "img_001.jpg"
        """
        path = self.image_paths[index]

        # PIL is used here because torchvision transforms expect PIL Images.
        from PIL import Image
        image = Image.open(path).convert("RGB")   # Ensure 3-channel RGB

        if self.transform:
            image = self.transform(image)

        return image, path.name


# ==============================================================================
# Per-model inference
# ==============================================================================

def run_model_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger
) -> tuple:
    """
    Run a single model on all images and return probabilities and filenames.

    Parameters
    ----------
    model : nn.Module
        A loaded, pretrained model with best.pth weights.
    dataloader : DataLoader
        DataLoader for the unlabelled images.
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    tuple:
        probs     : np.ndarray, shape (N, num_classes) — softmax probabilities
        filenames : list of str — image basenames in the same order as probs
    """
    model.eval()

    all_probs  = []
    filenames  = []

    with torch.no_grad():
        for images, names in dataloader:
            images  = images.to(device)
            outputs = model(images)

            # Convert logits to probabilities.
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            # names is a tuple of filename strings from the dataset's __getitem__.
            filenames.extend(list(names))

    return np.concatenate(all_probs, axis=0), filenames


# ==============================================================================
# Output formatting
# ==============================================================================

def build_output_json(
    filenames: list,
    fused_probs: np.ndarray,
    class_names: list,
    threshold: float,
    input_dir: str,
    fusion_method: str,
    timestamp: str
) -> dict:
    """
    Build the structured output JSON dictionary.

    Separates results into "flagged_images" (damaged, above threshold) and
    "all_predictions" (every image, for auditing).

    Parameters
    ----------
    filenames : list of str
        Image basenames.
    fused_probs : np.ndarray, shape (N, num_classes)
        Ensemble probability outputs.
    class_names : list of str
        Class names in label-index order.
    threshold : float
        Confidence threshold above which an image is flagged as damaged.
    input_dir : str
        Path to the input image directory (stored in run_info for traceability).
    fusion_method : str
        Name of the fusion method used.
    timestamp : str
        Run timestamp string.

    Returns
    -------
    dict
        Complete output structure ready to be saved as JSON.
    """
    # Index of the "undamaged" class — images predicted as undamaged are not
    # flagged unless they are predicted as a damage class above the threshold.
    # We identify the undamaged index by name to avoid hardcoding index 3.
    undamaged_index = class_names.index("undamaged") \
        if "undamaged" in class_names else -1

    flagged_images  = []
    all_predictions = []

    for i, filename in enumerate(filenames):
        probs         = fused_probs[i]
        predicted_idx = int(np.argmax(probs))
        confidence    = float(probs[predicted_idx])
        predicted_cls = class_names[predicted_idx]

        # An image is flagged if:
        #   (a) it is not predicted as undamaged, AND
        #   (b) the confidence exceeds the threshold.
        is_damaged  = predicted_idx != undamaged_index
        is_flagged  = is_damaged and confidence >= threshold

        # Build the full probability dict for this image.
        all_probs_dict = {
            cls: round(float(probs[j]), 4)
            for j, cls in enumerate(class_names)
        }

        prediction_record = {
            "filename":   filename,
            "prediction": predicted_cls,
            "confidence": round(confidence, 4),
            "flagged":    is_flagged,
        }
        all_predictions.append(prediction_record)

        if is_flagged:
            flagged_images.append({
                "filename":   filename,
                "prediction": predicted_cls,
                "confidence": round(confidence, 4),
                "all_probs":  all_probs_dict,
            })

    output = {
        "run_info": {
            "input_dir":     input_dir,
            "fusion_method": fusion_method,
            "threshold":     threshold,
            "total_images":  len(filenames),
            "flagged_count": len(flagged_images),
            "timestamp":     timestamp,
            "class_names":   class_names,
        },
        "flagged_images":  flagged_images,
        "all_predictions": all_predictions,
    }

    return output


# ==============================================================================
# Main deployment function
# ==============================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ------------------------------------------------------------------
    args      = parse_args()
    config    = load_config(args.config)

    class_names   = config["dataset"]["class_names"]
    num_classes   = config["dataset"]["num_classes"]
    fusion_method = config["ensemble"]["method"]
    threshold     = config["confidence_threshold"]
    input_sizes   = config["input_sizes"]

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    # ------------------------------------------------------------------
    # 2. Set up output directory and logging
    # ------------------------------------------------------------------
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir  = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a small log file alongside the output JSON.
    log_dir = output_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_run_logger(log_dir, __name__)
    logger.info(f"Deploy run — {timestamp}")
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Fusion method:    {fusion_method}")
    logger.info(f"Threshold:        {threshold}")

    # ------------------------------------------------------------------
    # 3. Detect device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 4. Run inference with each model
    # ------------------------------------------------------------------
    # Each model may require a different input size (AlexNet: 227, others: 224).
    # We create a separate DataLoader per model with the correct transform.
    # The filenames list is the same for all models since they process the
    # same images — we capture it from the first model and reuse it.

    probs_arrays   = []
    filenames_list = None

    for model_name, load_fn in MODEL_LOADERS.items():
        checkpoint_path = Path(config["saved_models"][model_name])

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found for {model_name}: {checkpoint_path}. "
                f"Skipping this model."
            )
            continue

        logger.info(f"Loading {model_name}...")

        # Load the model architecture and weights.
        model = load_fn(num_classes=num_classes, device=device)
        model = load_checkpoint(model, checkpoint_path, device)

        # Build the transform for this model's input size.
        input_size = input_sizes[model_name]
        transform  = get_eval_transforms(
            input_size = input_size,
            config     = {"normalisation": config.get("normalisation", {})}
        )

        # Build a DataLoader for this model.
        # We use UnlabelledImageDataset because there are no class subfolders.
        dataset    = UnlabelledImageDataset(input_dir, transform)
        dataloader = DataLoader(
            dataset,
            batch_size = 32,
            shuffle    = False,   # Must be False to keep image order consistent
            num_workers = 0
        )

        logger.info(f"  Running inference on {len(dataset)} images...")

        probs, filenames = run_model_inference(model, dataloader, device, logger)
        probs_arrays.append(probs)

        # Capture filenames from the first model — all models process
        # the same images in the same order (shuffle=False).
        if filenames_list is None:
            filenames_list = filenames

        logger.info(f"  {model_name} done.")

        # Free GPU memory between models — important on a GTX 1060 with 6GB.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(probs_arrays) < 2:
        logger.error(
            "At least 2 models must be available for ensemble inference. "
            "Check that saved_models paths in deploy_config.yaml are correct."
        )
        return

    # ------------------------------------------------------------------
    # 5. Apply ensemble fusion
    # ------------------------------------------------------------------
    if fusion_method not in ENSEMBLE_REGISTRY:
        logger.error(
            f"Unknown fusion method '{fusion_method}'. "
            f"Supported: {list(ENSEMBLE_REGISTRY.keys())}"
        )
        return

    logger.info(f"Applying fusion: {fusion_method}...")
    fusion_fn   = ENSEMBLE_REGISTRY[fusion_method]
    fused_probs = fusion_fn(probs_arrays)

    # ------------------------------------------------------------------
    # 6. Build and save the output JSON
    # ------------------------------------------------------------------
    logger.info("Building output JSON...")
    output = build_output_json(
        filenames     = filenames_list,
        fused_probs   = fused_probs,
        class_names   = class_names,
        threshold     = threshold,
        input_dir     = str(input_dir),
        fusion_method = fusion_method,
        timestamp     = timestamp
    )

    # Determine the output JSON path.
    if args.output:
        json_path = Path(args.output)
    else:
        json_name = config["output"]["filename"]
        json_path = log_dir / json_name

    save_json(output, json_path)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    total   = output["run_info"]["total_images"]
    flagged = output["run_info"]["flagged_count"]

    logger.info("=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"  Total images processed: {total}")
    logger.info(f"  Flagged as damaged:     {flagged} ({100*flagged/total:.1f}%)")
    logger.info(f"  Output JSON:            {json_path}")
    logger.info("=" * 60)

    # Print a quick preview of flagged images to the terminal.
    if flagged > 0:
        logger.info("Flagged images (top 10):")
        for item in output["flagged_images"][:10]:
            logger.info(
                f"  {item['filename']:<30} "
                f"{item['prediction']:<15} "
                f"conf: {item['confidence']:.3f}"
            )
        if flagged > 10:
            logger.info(f"  ... and {flagged - 10} more. See {json_path} for full list.")
    else:
        logger.info("No images flagged as damaged above the confidence threshold.")
        logger.info(f"Consider lowering 'confidence_threshold' in deploy_config.yaml "
                    f"(currently {threshold}).")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()
