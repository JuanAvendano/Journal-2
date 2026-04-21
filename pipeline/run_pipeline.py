"""
pipeline/run_pipeline.py
------------------------------------------------------------------------------
Master pipeline runner for all experiments (A, B, C).

How it works:
  1. Discovers all YAML configs under pipeline/configs/experiment_*/
  2. Runs each config sequentially via subprocess:
       python scripts/train.py --pipeline_config <yaml_path>
     Each run is a separate process — GPU memory is fully freed between runs.
  3. After all models for a dataset group finish (e.g. A1_VGG16, A1_AlexNet,
     A1_ResNet50), runs ensemble evaluation automatically.
  4. Skips and logs any failed run — pipeline continues regardless.
  5. Calls report_generator.py at the very end.

Usage (run from the repo root — Journal-2/):
  python pipeline/run_pipeline.py                 # All experiments
  python pipeline/run_pipeline.py --group A       # Only group A
  python pipeline/run_pipeline.py --group A B     # Groups A and B

Expected folder structure:
  Journal-2/
    pipeline/
      run_pipeline.py           <- this file
      report_generator.py
      configs/
        experiment_A/
          A1_VGG16.yaml
          A1_AlexNet.yaml
          A1_ResNet50.yaml
          A2_VGG16.yaml
          ... (12 files total for group A)
        experiment_B/
          B1_VGG16.yaml
          ... (4 files)
        experiment_C/
          C_crack_VGG16.yaml
          ... (9 files)
    scripts/
      train.py                  <- called by this script
    results/
      training/                 <- where train.py writes outputs
        vgg16/
          A1_VGG16/             <- named by experiment_id, not timestamp
            metrics/test_metrics.json
            predictions/test_predictions.csv
            ...
"""

import subprocess
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml


# ==============================================================================
# Paths — all relative to the repo root (Journal-2/)
# ==============================================================================
# These paths assume you run this script from the repo root:
#   cd Journal-2
#   python pipeline/run_pipeline.py

# Repo root is the parent of the pipeline/ folder
REPO_ROOT = Path(__file__).resolve().parent.parent

# Training script
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train(1).py"

# Ensemble evaluation script — adjust if yours is named differently
ENSEMBLE_SCRIPT = REPO_ROOT / "scripts" / "ensemble_eval.py"

# Report generator — same folder as this file
REPORT_SCRIPT = Path(__file__).parent / "report_generator.py"

# Where train.py writes its outputs — results/training/
RESULTS_TRAINING_DIR = REPO_ROOT / "results" / "training"

# Pipeline config folder
CONFIGS_DIR = Path(__file__).parent / "configs"

# Pipeline log folder — kept separate from training results
PIPELINE_LOGS_DIR = REPO_ROOT / "results" / "pipeline_logs"

# Python interpreter — always the same one running this script
PYTHON = sys.executable


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging() -> logging.Logger:
    """
    Create a logger that writes to both the console and a timestamped log file.
    The log is stored in results/pipeline_logs/ so it doesn't clutter pipeline/.
    """
    PIPELINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = PIPELINE_LOGS_DIR / f"pipeline_{timestamp}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Pipeline log: {log_path}")
    return logger


# ==============================================================================
# Config discovery
# ==============================================================================

def discover_configs(groups: list | None) -> dict:
    """
    Walk pipeline/configs/ and return a nested structure:

        {
            "A": {
                "A1": [Path("A1_VGG16.yaml"), Path("A1_AlexNet.yaml"), ...],
                "A2": [...],
            },
            "B": { "B1": [...], "B2": [...] },
            "C": { "C_crack": [...], "C_eff": [...], "C_spall": [...] },
        }

    The dataset_id is everything in the filename before the model name
    (the last underscore-separated token):
        A1_VGG16.yaml       -> dataset_id = "A1"
        C_crack_VGG16.yaml  -> dataset_id = "C_crack"

    Parameters
    ----------
    groups : list | None
        Group letters to include e.g. ["A", "B"]. None means all groups.
    """
    result = {}

    for group_dir in sorted(CONFIGS_DIR.iterdir()):
        if not group_dir.is_dir():
            continue

        # "experiment_A" -> "A"
        group_letter = group_dir.name.split("_")[-1].upper()

        if groups and group_letter not in groups:
            continue

        result[group_letter] = {}

        for config_file in sorted(group_dir.glob("*.yaml")):
            # Split filename into parts and use everything except the last
            # (model name) as the dataset_id.
            # "A1_VGG16"      -> ["A1", "VGG16"]      -> dataset_id = "A1"
            # "C_crack_VGG16" -> ["C", "crack", "VGG16"] -> dataset_id = "C_crack"
            parts      = config_file.stem.split("_")
            dataset_id = "_".join(parts[:-1])

            if dataset_id not in result[group_letter]:
                result[group_letter][dataset_id] = []
            result[group_letter][dataset_id].append(config_file)

    return result


# ==============================================================================
# Single training run
# ==============================================================================

def run_training(config_path: Path, logger: logging.Logger) -> tuple:
    """
    Execute one training run via subprocess.

    Calls:
        python scripts/train.py --pipeline_config <config_path>

    Because train.py uses the experiment_id from the YAML as the run folder
    name, the output lands in a predictable location:
        results/training/{model}/{experiment_id}/

    Returns
    -------
    (success: bool, experiment_id: str | None)
    """
    # Read experiment_id from YAML before starting so we can log it clearly
    try:
        with open(config_path) as f:
            pipeline_cfg = yaml.safe_load(f)
        experiment_id = pipeline_cfg["experiment_id"]
        model_name    = pipeline_cfg["model"]
    except Exception as e:
        logger.error(f"  Could not read {config_path.name}: {e}. Skipping.")
        return False, None

    logger.info(f"  Starting: {experiment_id}  (model={model_name})")
    start = time.time()

    try:
        subprocess.run(
            [PYTHON, str(TRAIN_SCRIPT),
             "--pipeline_config", str(config_path)],
            check=True,           # Raises CalledProcessError on non-zero exit
            capture_output=False  # Let stdout/stderr stream live to console
        )
        elapsed = time.time() - start
        logger.info(f"  Done:    {experiment_id} — {elapsed / 60:.1f} min  OK")
        return True, experiment_id

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logger.error(
            f"  FAILED:  {experiment_id} — {elapsed / 60:.1f} min "
            f"(exit code {e.returncode}). Skipping."
        )
        return False, experiment_id

    except Exception as e:
        logger.error(f"  ERROR:   {experiment_id} — {e}. Skipping.")
        return False, experiment_id


# ==============================================================================
# Ensemble evaluation
# ==============================================================================

def run_ensemble(dataset_id: str,
                 completed_ids: list,
                 logger: logging.Logger) -> None:
    """
    Run ensemble evaluation after all individual models for a dataset finish.

    Passes the experiment_ids of completed runs to ensemble_eval.py so it can
    locate their predictions files inside results/training/{model}/{exp_id}/.

    Skipped if fewer than 2 models completed.

    Parameters
    ----------
    dataset_id    : e.g. "A1"
    completed_ids : e.g. ["A1_VGG16", "A1_AlexNet", "A1_ResNet50"]
    logger        : Pipeline logger.
    """
    if len(completed_ids) < 2:
        logger.warning(
            f"  Ensemble skipped for {dataset_id}: "
            f"only {len(completed_ids)} model(s) completed — need at least 2."
        )
        return

    if not ENSEMBLE_SCRIPT.exists():
        logger.warning(
            f"  Ensemble script not found at {ENSEMBLE_SCRIPT}. "
            f"Skipping ensemble for {dataset_id}."
        )
        return

    output_id = f"{dataset_id}_ensemble"
    logger.info(
        f"  Ensemble: {output_id}  [{', '.join(completed_ids)}]"
    )

    try:
        subprocess.run(
            [
                PYTHON, str(ENSEMBLE_SCRIPT),
                "--run_ids",     *completed_ids,
                "--results_dir", str(RESULTS_TRAINING_DIR),
                "--output_id",   output_id,
            ],
            check=True,
            capture_output=False
        )
        logger.info(f"  Ensemble {output_id} complete  OK")

    except subprocess.CalledProcessError as e:
        logger.error(f"  Ensemble {output_id} FAILED (exit {e.returncode}).")
    except Exception as e:
        logger.error(f"  Ensemble {output_id} ERROR: {e}.")


# ==============================================================================
# Main pipeline loop
# ==============================================================================

def run_pipeline(groups: list | None) -> None:
    """
    Iterate over experiment groups, dataset IDs, and model configs, running
    each training job in sequence and triggering ensemble evaluation after
    each complete dataset group finishes.
    """
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info(f"Groups    : {groups if groups else 'ALL'}")
    logger.info(f"Train     : {TRAIN_SCRIPT}")
    logger.info(f"Results   : {RESULTS_TRAINING_DIR}")
    logger.info("=" * 60)

    if not TRAIN_SCRIPT.exists():
        logger.error(
            f"Training script not found: {TRAIN_SCRIPT}\n"
            "Make sure you run from the repo root: cd Journal-2"
        )
        return

    all_configs = discover_configs(groups)

    if not all_configs:
        logger.error(
            f"No YAML configs found under {CONFIGS_DIR}.\n"
            "Run:  python pipeline/generate_configs.py"
        )
        return

    total_runs = total_success = total_failed = 0
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Outer loop: experiment groups  (A, B, C)
    # ------------------------------------------------------------------
    for group, datasets in all_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"GROUP {group}  —  {len(datasets)} dataset(s)")
        logger.info(f"{'='*60}")

        # --------------------------------------------------------------
        # Inner loop: dataset IDs within a group
        # --------------------------------------------------------------
        for dataset_id, config_files in datasets.items():
            logger.info(
                f"\n  --- {dataset_id}  ({len(config_files)} config(s)) ---"
            )

            completed_ids = []   # Collect experiment_ids that succeeded

            for config_path in config_files:
                total_runs += 1
                success, exp_id = run_training(config_path, logger)

                if success:
                    total_success += 1
                    completed_ids.append(exp_id)
                else:
                    total_failed += 1

            # After all models for this dataset are done, run ensemble
            logger.info("")
            run_ensemble(dataset_id, completed_ids, logger)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Generating final report...")
    logger.info(f"{'='*60}")

    try:
        subprocess.run(
            [PYTHON, str(REPORT_SCRIPT),
             "--results_dir", str(RESULTS_TRAINING_DIR)],
            check=True
        )
        logger.info("Report done  OK")
    except Exception as e:
        logger.error(f"Report failed: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_h = (time.time() - pipeline_start) / 3600
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE in {elapsed_h:.2f} hours")
    logger.info(f"  Total  : {total_runs}")
    logger.info(f"  OK     : {total_success}")
    logger.info(f"  Failed : {total_failed}")
    logger.info(f"{'='*60}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all CNN training experiments sequentially."
    )
    parser.add_argument(
        "--group",
        nargs="+",
        default=None,
        help="Group(s) to run: A, B, C. Omit to run all."
    )
    args = parser.parse_args()

    groups = [g.upper() for g in args.group] if args.group else None
    run_pipeline(groups)
