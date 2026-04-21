"""
pipeline/generate_configs.py
------------------------------------------------------------------------------
Auto-generates all 25 pipeline YAML config files for experiments A, B, and C.

Run once from the repo root before starting the pipeline:
    python pipeline/generate_configs.py

Output:
    pipeline/configs/experiment_A/   (12 files: A1-A4 x VGG16/AlexNet/ResNet50)
    pipeline/configs/experiment_B/   ( 4 files: B1-B2 x VGG16/AlexNet)
    pipeline/configs/experiment_C/   ( 9 files: 3 configs x 3 models)

THINGS TO ADJUST BEFORE RUNNING:
  1. BASE_CONFIG_PATH  — path to your existing train_config.yaml
  2. The dataset_path values in experiment_A, B, and C definitions below.
     These must match the actual folder paths where you built each dataset.
  3. TEST_SET_PATH and BINARY_TEST_SET_PATH — your fixed test sets.
"""

from pathlib import Path
import yaml


# ==============================================================================
# Paths — adjust these to match your actual dataset locations
# ==============================================================================

# Your existing base config (used as reference — not overridden by generate_configs)
BASE_CONFIG_PATH = r"C:/Users/jcac/OneDrive - KTH/Python/CNN/Journal-2/configs/train_config.yaml"

# Subfolder names used by your dataset creation script inside each dataset root.
# For example, if A1/ contains "train/" and "val/" subfolders, leave these as-is.
# If your script creates "01-train/" and "02-validation/", change them here.
TRAIN_SUBFOLDER = "01-train"
VAL_SUBFOLDER   = "02-validation"

# Fixed 4-class test set (800 images: 200 per class). Never changes.
TEST_SET_PATH = r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/Test/"

# Fixed binary test set for Experiment B (600 images: 300 crack + 300 undamaged)
BINARY_TEST_SET_PATH = r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/B/B1/"

# Where configs are written
CONFIGS_ROOT = Path(__file__).parent / "configs"

# Class names for 4-class and binary tasks
CLASS_NAMES_4  = ["crack", "efflorescence", "spalling", "undamaged"]
CLASS_NAMES_2  = ["crack", "undamaged"]


# ==============================================================================
# Experiment definitions
# ==============================================================================

# --- Experiment A: Balanced scaling -------------------------------------------
# 4 dataset sizes x 3 models = 12 configs
# ResNet50 gets 70 epochs (needs more to converge); others get 50.

EXPERIMENT_A = {
    "group":   "A",
    "subdir":  "experiment_A",
    "datasets": [
        {
            "id":            "A1",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/A/A1/",   # 500 per class, 80/20 split
            "per_class":     500,
        },
        {
            "id":            "A2",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/A/A2/",   # 1000 per class
            "per_class":     1000,
        },
        {
            "id":            "A3",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/A/A3/",   # 2000 per class
            "per_class":     2000,
        },
        {
            "id":            "A4",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/A/A4/",   # 3500 per class
            "per_class":     3500,
        },
    ],
    "models": [
        {"name": "vgg16",    "epochs": 50},
        {"name": "alexnet",  "epochs": 50},
        {"name": "resnet50", "epochs": 70},  # ResNet needs more epochs
    ],
    "num_classes":   4,
    "class_names":   CLASS_NAMES_4,
    "class_weights": [1.0, 1.0, 1.0, 1.0],  # No weighting in Experiment A
    "test_set_path": TEST_SET_PATH,
    "batch_size":    32,
    "learning_rate": 0.0001,
    "random_seed":   42,
}

# --- Experiment B: Binary overfitting diagnostic ------------------------------
# 2 dataset sizes x 2 models = 4 configs
# Only VGG16 and AlexNet — the two overfitters.
# ResNet50 is excluded because it does not show the same overfitting pattern.

EXPERIMENT_B = {
    "group":   "B",
    "subdir":  "experiment_B",
    "datasets": [
        {
            "id":            "B1",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/B/B1/",   # 1000 per class (crack + undamaged)
            "per_class":     1000,
        },
        {
            "id":            "B2",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/B//B2/",   # 5000 per class
            "per_class":     5000,
        },
    ],
    "models": [
        {"name": "vgg16",   "epochs": 50},
        {"name": "alexnet", "epochs": 50},
    ],
    "num_classes":   2,
    "class_names":   CLASS_NAMES_2,
    "class_weights": [1.0, 1.0],
    "test_set_path": BINARY_TEST_SET_PATH,
    "batch_size":    32,
    "learning_rate": 0.0001,
    "random_seed":   42,
}

# --- Experiment C: Specialist training via class-weighted loss ----------------
# 3 specialist configs x 3 models = 9 configs
# All reuse dataset A3 (2000 per class) — no new data preparation needed.
# Each specialist config sets the target class weight to 3.0 and others to 1.0.
#
# class_weights order must match class_names order:
#   [crack, efflorescence, spalling, undamaged]

EXPERIMENT_C = {
    "group":   "C",
    "subdir":  "experiment_C",
    "datasets": [
        {
            "id":            "C_crack",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/C/C1/",   # Reuse A3 — no new dataset needed
            # crack is index 0 -> weight 3.0; others 1.0
            "class_weights": [3.0, 1.0, 1.0, 1.0],
        },
        {
            "id":            "C_efflorescence",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/C/C1/",
            # efflorescence is index 1 -> weight 3.0
            "class_weights": [1.0, 3.0, 1.0, 1.0],
        },
        {
            "id":            "C_spalling",
            "dataset_path":  r"D:/JCA/07-Data/01_Concrete/03-experiment_datasets/C/C1/",
            # spalling is index 2 -> weight 3.0
            "class_weights": [1.0, 1.0, 3.0, 1.0],
        },
    ],
    "models": [
        {"name": "vgg16",    "epochs": 50},
        {"name": "alexnet",  "epochs": 50},
        {"name": "resnet50", "epochs": 70},
    ],
    "num_classes":   4,
    "class_names":   CLASS_NAMES_4,
    "test_set_path": TEST_SET_PATH,
    "batch_size":    32,
    "learning_rate": 0.0001,
    "random_seed":   42,
}


# ==============================================================================
# Config generation
# ==============================================================================

def build_config(experiment_id: str,
                 group: str,
                 model_name: str,
                 train_path: str,
                 val_path: str,
                 test_set_path: str,
                 num_classes: int,
                 class_names: list,
                 class_weights: list,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 random_seed: int) -> dict:
    """
    Build a flat dictionary representing one pipeline YAML config.

    The keys here are what train.py reads in apply_pipeline_config():
      train_path    -> config["paths"]["train"]
      val_path      -> config["paths"]["val"]
      test_set_path -> config["paths"]["test"]

    train_path and val_path are derived in generate_experiment() by appending
    the subfolder names to dataset_path. If your dataset creation script uses
    different subfolder names (e.g. "01-train" instead of "train"), update
    TRAIN_SUBFOLDER and VAL_SUBFOLDER at the top of this file.
    """
    return {
        "experiment_id":  experiment_id,
        "group":          group,
        "model":          model_name,
        "train_path":     train_path,
        "val_path":       val_path,
        "test_set_path":  test_set_path,
        "num_classes":    num_classes,
        "class_names":    class_names,
        "class_weights":  class_weights,
        "epochs":         epochs,
        "batch_size":     batch_size,
        "learning_rate":  learning_rate,
        "random_seed":    random_seed,
    }


def generate_experiment(exp_def: dict) -> list[Path]:
    """
    Generate all YAML files for one experiment definition dict.

    Returns a list of Paths for the files created.
    """
    output_dir = CONFIGS_ROOT / exp_def["subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []

    for dataset in exp_def["datasets"]:
        for model_def in exp_def["models"]:

            model_name    = model_def["name"]
            epochs        = model_def["epochs"]
            dataset_id    = dataset["id"]
            experiment_id = f"{dataset_id}_{model_name.upper()}"

            # Per-dataset class_weights override experiment-level weights
            # (used in Experiment C where each dataset has different weights)
            weights = dataset.get("class_weights", exp_def.get("class_weights"))

            # Derive train and val paths from the dataset root folder.
            # The trailing slash matches the style in train_config.yaml.
            dataset_root = dataset["dataset_path"].rstrip("/") + "/"
            train_path   = dataset_root + TRAIN_SUBFOLDER + "/"
            val_path     = dataset_root + VAL_SUBFOLDER   + "/"

            cfg = build_config(
                experiment_id  = experiment_id,
                group          = exp_def["group"],
                model_name     = model_name,
                train_path     = train_path,
                val_path       = val_path,
                test_set_path  = exp_def["test_set_path"],
                num_classes    = exp_def["num_classes"],
                class_names    = exp_def["class_names"],
                class_weights  = weights,
                epochs         = epochs,
                batch_size     = exp_def["batch_size"],
                learning_rate  = exp_def["learning_rate"],
                random_seed    = exp_def["random_seed"],
            )

            # YAML filename: e.g. "A1_VGG16.yaml"
            filename = output_dir / f"{experiment_id}.yaml"

            with open(filename, "w") as f:
                # default_flow_style=False forces block style (human-readable)
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            created.append(filename)
            print(f"  Created: {filename.relative_to(CONFIGS_ROOT.parent)}")

    return created


def main():
    print("Generating pipeline YAML configs.../n")

    all_created = []
    for exp_def in [EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_C]:
        print(f"--- Experiment {exp_def['group']} ---")
        created = generate_experiment(exp_def)
        all_created.extend(created)
        print()

    print(f"Done. {len(all_created)} config files created under pipeline/configs/")
    print()
    print("Next steps:")
    print("  1. Check data paths in each YAML match your actual dataset folders.")
    print("  2. Run:  python pipeline/run_pipeline.py --group A")
    print("           (test group A first before running everything overnight)")


if __name__ == "__main__":
    main()
