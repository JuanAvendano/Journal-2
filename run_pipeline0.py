"""
run_pipeline0.py
------------------------------------------------------------------------------
Pipeline runner — trains CNN models sequentially and runs ensemble evaluation.

Edit the configuration section below to control what the pipeline does.
No command-line arguments are needed for normal use — just set the options
here and run:

    python run_pipeline0.py

The pipeline delegates to the existing scripts:
    scripts/train.py      ← handles one model training run
    scripts/ensemble_eval.py   ← handles ensemble evaluation

Each script runs in its own subprocess, which means GPU memory is fully
released between models. This is important when training multiple large
models sequentially on a GPU with limited memory (e.g. GTX 1060 6GB).
"""

import subprocess
import sys
import time
from datetime import datetime


# ==============================================================================
# CONFIGURATION — edit this section before running
# ==============================================================================

# ------------------------------------------------------------------------------
# Mode
# ------------------------------------------------------------------------------
# Controls which phases of the pipeline run.
#   "full"          : train all enabled models, then run ensemble evaluation
#   "train_only"    : train models only, skip ensemble
#   "ensemble_only" : skip training, run ensemble on existing predictions
MODE = "full"

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
# Set enabled=True/False to include or exclude a model from training.
# Order matters — models are trained in the order listed here.
MODELS = [
    {"name": "vgg16",    "enabled": True},
    {"name": "resnet50", "enabled": True},
    {"name": "alexnet",  "enabled": True},
]

# ------------------------------------------------------------------------------
# Config paths
# ------------------------------------------------------------------------------
TRAIN_CONFIG    = "configs/train_config.yaml"
ENSEMBLE_CONFIG = "configs/ensemble_config.yaml"

# ------------------------------------------------------------------------------
# Behaviour
# ------------------------------------------------------------------------------
# If True, the pipeline continues to the next model if one fails.
# If False, any failure stops the pipeline immediately.
CONTINUE_ON_FAILURE = True

# If True, trains and evaluates the MLP meta-learner in the ensemble step.
TRAIN_MLP = False

# If True, plots are displayed interactively in addition to being saved.
SHOW_PLOTS = False


# ==============================================================================
# Pipeline logic — no need to edit below this line
# ==============================================================================

def run_command(cmd: list, label: str) -> bool:
    """
    Run a command in a subprocess and return whether it succeeded.

    Parameters
    ----------
    cmd : list of str
        The command to run, e.g. [sys.executable, "scripts/train.py", ...].
        sys.executable ensures the same Python interpreter (and virtual
        environment) that is running this script is used for the subprocess.
    label : str
        Human-readable label for logging, e.g. "train vgg16".

    Returns
    -------
    bool
        True if the command succeeded (exit code 0), False if it failed.
    """
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()

    # subprocess.run() runs the command and waits for it to finish.
    # check=False means we handle the return code ourselves rather than
    # raising an exception automatically — this lets us log the failure
    # and decide whether to continue.
    result = subprocess.run(cmd, check=False)

    duration = time.time() - start
    minutes  = duration / 60

    if result.returncode == 0:
        print(f"\n{'='*60}")
        print(f"  DONE: {label} ({minutes:.1f} min)")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print(f"  FAILED: {label} (exit code {result.returncode}, "
              f"{minutes:.1f} min)")
        print(f"{'='*60}\n")
        return False


def run_ensemble():
    """Build and run the ensemble_eval.py command."""
    cmd = [
        sys.executable, "scripts/ensemble_eval.py",
        "--config", ENSEMBLE_CONFIG,
    ]
    if TRAIN_MLP:
        cmd.append("--train_mlp")
    if SHOW_PLOTS:
        cmd.append("--show_plots")

    run_command(cmd, label="ensemble evaluation")


def main():
    pipeline_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  PIPELINE STARTING")
    print(f"  Mode:      {MODE}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # ------------------------------------------------------------------
    # Training phase
    # ------------------------------------------------------------------
    training_results = {}   # model_name → True/False/None (None = skipped)

    if MODE in ("full", "train_only"):
        print("--- TRAINING PHASE ---\n")

        for model_cfg in MODELS:
            name    = model_cfg["name"]
            enabled = model_cfg.get("enabled", True)

            if not enabled:
                print(f"  Skipping {name} (disabled in MODELS list).\n")
                training_results[name] = None
                continue

            cmd = [
                sys.executable, "scripts/train.py",
                "--model",  name,
                "--config", TRAIN_CONFIG,
            ]
            if SHOW_PLOTS:
                cmd.append("--show_plots")

            success = run_command(cmd, label=f"train {name}")
            training_results[name] = success

            if not success and not CONTINUE_ON_FAILURE:
                print("CONTINUE_ON_FAILURE is False — stopping pipeline.")
                break

    # ------------------------------------------------------------------
    # Ensemble phase
    # ------------------------------------------------------------------
    if MODE in ("full", "ensemble_only"):

        if MODE == "full":
            # Check that at least 2 models trained successfully before
            # attempting the ensemble.
            successful = [n for n, s in training_results.items() if s is True]
            if len(successful) < 2:
                print(
                    f"\nOnly {len(successful)} model(s) trained successfully "
                    f"({successful}). Need at least 2 for ensemble. "
                    f"Skipping ensemble step.\n"
                )
            else:
                print(f"\n--- ENSEMBLE PHASE ---")
                print(f"  Using models: {successful}\n")
                run_ensemble()
        else:
            # ensemble_only mode — no training results to check.
            print("\n--- ENSEMBLE PHASE ---\n")
            run_ensemble()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_min = (time.time() - pipeline_start) / 60

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE ({total_min:.1f} min total)")
    print(f"{'#'*60}\n")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()