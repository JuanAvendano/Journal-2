"""
src/utils/io_utils.py
------------------------------------------------------------------------------
Utility functions for all file input/output operations in the repository.

This module centralises every read/write operation so that:
  - No other file needs to know the exact CSV column layout or JSON structure.
  - If the file format ever changes, you only fix it in one place.
  - Path and folder creation logic is not scattered across scripts.

Nothing in this file trains models or computes metrics — it only handles data
movement between disk and Python objects.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


# ==============================================================================
# Run directory management
# ==============================================================================

def make_run_dir(base_dir: str, model_or_method_name: str) -> Path:
    """
    Create and return a timestamped directory for a single training or
    evaluation run.

    A new folder is created each time you run training or evaluation, named
    after the current date and time. This means results from different runs
    never overwrite each other.

    Example output path:
        results/training/vgg16/2026-03-19_14-32/

    Parameters
    ----------
    base_dir : str
        The parent directory, e.g. "results/training" or "results/ensemble".
    model_or_method_name : str
        Subfolder name, e.g. "vgg16" or the run label.

    Returns
    -------
    Path
        The full path to the newly created run directory.
    """
    # datetime.now() gives the current date and time.
    # strftime() formats it as a readable string, e.g. "2026-03-19_14-32".
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Path(...) / "subfolder" is Python's cross-platform way of joining paths.
    # It works on both Windows and Linux without needing to worry about
    # forward vs backslashes.
    run_dir = Path(base_dir) / model_or_method_name / timestamp

    # exist_ok=True means no error is raised if the folder already exists.
    # parents=True means intermediate folders are created if they don't exist.
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_latest_run_dir(base_dir: str, model_name: str) -> Path:
    """
    Return the most recently created run directory for a given model.

    This is what the keyword "latest" resolves to in ensemble_config.yaml.
    It looks at all timestamped subfolders and returns the one with the
    most recent name (timestamps sort lexicographically, so the last one
    alphabetically is also the most recent).

    Parameters
    ----------
    base_dir : str
        e.g. "results/training"
    model_name : str
        e.g. "vgg16"

    Returns
    -------
    Path
        Path to the most recent run folder.

    Raises
    ------
    FileNotFoundError
        If no run folders exist yet for this model.
    """
    model_dir = Path(base_dir) / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No results found for model '{model_name}' in '{base_dir}'. "
            f"Have you trained this model yet?"
        )

    # List all subdirectories, sort them (timestamps sort correctly as strings),
    # and take the last one.
    run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(
            f"No run folders found in '{model_dir}'."
        )

    return run_dirs[-1]


def resolve_prediction_path(path_str: str, base_dir: str, model_name: str) -> Path:
    """
    Resolve a prediction CSV path from ensemble_config.yaml.

    If the path contains the keyword "latest", it is replaced with the
    actual path of the most recent run. Otherwise the path is used as-is.

    Parameters
    ----------
    path_str : str
        The path string from the config, e.g.
        "results/training/vgg16/latest/predictions.csv"
    base_dir : str
        Base results directory, e.g. "results/training".
    model_name : str
        Model name, e.g. "vgg16".

    Returns
    -------
    Path
        Resolved absolute path to the CSV file.
    """
    if "latest" in path_str:
        latest_dir = get_latest_run_dir(base_dir, model_name)
        # Replace the "latest" segment with the actual timestamped folder name.
        resolved = Path(path_str.replace(
            f"{model_name}/latest", f"{model_name}/{latest_dir.name}"
        ))
    else:
        resolved = Path(path_str)

    if not resolved.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {resolved}")

    return resolved


# ==============================================================================
# CSV operations
# ==============================================================================

def save_predictions_csv(
    save_dir: Path,
    filename: str,
    image_names: list,
    true_labels: list,
    predicted_labels: list,
    class_probs,          # numpy array of shape (N, num_classes)
    class_names: list
) -> None:
    """
    Save model predictions and class probabilities to a CSV file.

    This is the standardised format that all three model training scripts
    produce, and that all ensemble methods read. Having one format means
    the ensemble code does not need to know which model produced the file.

    Output CSV columns:
        Image_Name | class_0_prob | class_1_prob | ... | True_Label | Predicted_Label

    Parameters
    ----------
    save_dir : Path
        Directory to save the CSV into.
    filename : str
        CSV filename, e.g. "predictions.csv".
    image_names : list of str
        Filenames of the images (just the basename, not full path).
    true_labels : list of int
        Ground truth class indices.
    predicted_labels : list of int
        Model predicted class indices.
    class_probs : np.ndarray, shape (N, num_classes)
        Softmax probabilities for each class.
    class_names : list of str
        Class names, used as column headers.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build a DataFrame column by column for clarity.
    # Using the actual class names as column headers (e.g. "crack_prob") is
    # more readable than generic "Class_0_prob".
    prob_columns = {f"{name}_prob": class_probs[:, i]
                    for i, name in enumerate(class_names)}

    df = pd.DataFrame(prob_columns)
    df.insert(0, "Image_Name", image_names)   # Insert as the first column
    df["True_Label"] = true_labels
    df["Predicted_Label"] = predicted_labels

    csv_path = save_dir / filename
    df.to_csv(csv_path, index=False)
    print(f"  Predictions saved to: {csv_path}")


def load_predictions_csv(csv_path: Path, class_names: list) -> dict:
    """
    Load a model predictions CSV file into a structured dictionary.

    This is the single loader used by all ensemble methods, replacing the
    multiple versions of parse_custom_csv() that existed previously.
    It handles the CSV format produced by save_predictions_csv() above.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    class_names : list of str
        Class names to extract probability columns for.

    Returns
    -------
    dict with keys:
        "image_names"  : list of str
        "true_labels"  : list of int
        "predictions"  : list of int
        "probs"        : np.ndarray of shape (N, num_classes)
    """
    df = pd.read_csv(csv_path)

    # Build the expected column names to extract probabilities.
    prob_cols = [f"{name}_prob" for name in class_names]

    # Validate that the expected columns are present before trying to read them.
    missing = [col for col in prob_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file '{csv_path}' is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return {
        "image_names": df["Image_Name"].tolist(),
        "true_labels": df["True_Label"].tolist(),
        "predictions": df["Predicted_Label"].tolist(),
        # .values converts the DataFrame columns to a numpy array (N, num_classes)
        "probs":       df[prob_cols].values,
    }


def save_metrics_csv(save_dir: Path, filename: str, metrics_list: list) -> None:
    """
    Save a list of per-method metric dictionaries to a CSV summary file.

    Parameters
    ----------
    save_dir : Path
        Directory to save into.
    filename : str
        e.g. "metrics_summary.csv"
    metrics_list : list of dict
        Each dict should have keys like "Method", "Accuracy", "F1 Score", etc.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    path = save_dir / filename
    df.to_csv(path, index=False)
    print(f"  Metrics summary saved to: {path}")


# ==============================================================================
# JSON operations
# ==============================================================================

def save_json(data: dict, save_path: Path) -> None:
    """
    Save a Python dictionary to a JSON file.

    Used for saving training metrics, ensemble results, and deployment output.
    JSON is preferred over CSV for structured data with nested fields.

    Parameters
    ----------
    data : dict
        Data to serialise.
    save_path : Path
        Full file path including filename, e.g. run_dir / "metrics.json".
    """
    # Ensure the parent directory exists.
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # indent=4 makes the JSON file human-readable with 4-space indentation.
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"  JSON saved to: {save_path}")


def load_json(load_path: Path) -> dict:
    """
    Load a JSON file and return it as a Python dictionary.

    Parameters
    ----------
    load_path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        The loaded data.
    """
    with open(load_path, "r") as f:
        data = json.load(f)
    return data


# ==============================================================================
# Config loading
# ==============================================================================

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return it as a Python dictionary.

    PyYAML (imported as 'yaml') parses the YAML syntax into native Python
    types: strings, lists, dicts, booleans, numbers — so you can access
    values like config["training"]["batch_size"] directly.

    Parameters
    ----------
    config_path : str
        Path to the .yaml config file.

    Returns
    -------
    dict
        The full config as a nested dictionary.
    """
    # yaml is not in the standard library — it comes from the PyYAML package.
    # We import it here rather than at the top of the file so that the rest of
    # io_utils still works even if PyYAML is not installed (useful for testing).
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # safe_load prevents arbitrary code execution

    return config
