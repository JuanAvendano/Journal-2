"""
src/utils/logger.py
------------------------------------------------------------------------------
Logging setup for the repository.

Python's built-in 'logging' module is more powerful than using print()
statements because it:
  - Automatically adds timestamps to every message.
  - Lets you control verbosity (DEBUG, INFO, WARNING, ERROR) in one place.
  - Writes to both the console and a log file simultaneously.
  - Tags each message with which module it came from.

Usage in any other file:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)       # __name__ is the current module name
    logger.info("Starting training...")
    logger.warning("Low GPU memory.")
    logger.error("File not found.")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Create and return a logger for a given module.

    Each module in the codebase calls get_logger(__name__) to get its own
    logger. The __name__ variable in Python is automatically set to the
    full module path (e.g. "src.models.base_model"), so log messages are
    clearly tagged with their source.

    Parameters
    ----------
    name : str
        The logger name, typically passed as __name__ from the calling module.
    log_file : Path, optional
        If provided, log messages are also written to this file in addition
        to the console. Pass the run directory's log file path from train.py.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # getLogger() returns the same logger object if called multiple times with
    # the same name — so calling get_logger(__name__) twice in the same module
    # does not create duplicate handlers.
    logger = logging.getLogger(name)

    # Only configure the logger if it has no handlers yet.
    # This prevents duplicate log lines if get_logger() is called more than once.
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)   # Capture everything from DEBUG upwards

        # ------------------------------------------------------------------
        # Formatter: defines what each log line looks like
        # ------------------------------------------------------------------
        # %(asctime)s   → timestamp, e.g. "2026-03-19 14:32:01"
        # %(name)s      → logger name (module path)
        # %(levelname)s → DEBUG / INFO / WARNING / ERROR
        # %(message)s   → the actual log message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # ------------------------------------------------------------------
        # Console handler: prints to the terminal
        # ------------------------------------------------------------------
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)   # Show INFO and above in terminal
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ------------------------------------------------------------------
        # File handler: writes to a .log file (optional)
        # ------------------------------------------------------------------
        if log_file is not None:
            # Make sure the directory exists before trying to write the file.
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a")  # "a" = append
            file_handler.setLevel(logging.DEBUG)   # Write everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _log_config(logger: logging.Logger, data: dict, indent: int = 0) -> None:
    """
    Recursively log all key-value pairs from a config dictionary.

    Nested dictionaries are indented so the hierarchy is easy to read
    in the log file. For example, a config like:
        {"training": {"lr": 0.001, "epochs": 50}, "model": "VGG16"}
    will be logged as:
        training:
          lr: 0.001
          epochs: 50
        model: VGG16

    Parameters
    ----------
    logger : logging.Logger
        The logger to write to.
    data : dict
        The config dictionary (or sub-dictionary during recursion).
    indent : int
        Current indentation depth (increases by 1 for each nested level).
    """
    prefix = "  " * indent  # Two spaces per nesting level
    for key, value in data.items():
        if isinstance(value, dict):
            # Nested section — log the key as a header, then recurse
            logger.info(f"{prefix}{key}:")
            _log_config(logger, value, indent + 1)
        else:
            logger.info(f"{prefix}{key}: {value}")


def get_run_logger(run_dir: Path, module_name: str, config: dict = None) -> logging.Logger:
    """
    Convenience function that creates a logger and automatically saves its
    output to a 'run.log' file inside the run directory.

    This is the function called by scripts/train.py and scripts/ensemble_eval.py
    so that every run has a complete log file saved alongside its results.

    If a config dictionary is provided, its full contents are written at the
    top of the log file. This makes every log file self-contained: you can
    always trace back which parameters produced a given result, even after
    modifying the config file multiple times.

    Parameters
    ----------
    run_dir : Path
        The timestamped run directory created by io_utils.make_run_dir().
    module_name : str
        Typically passed as __name__ from the calling script.
    config : dict, optional
        The loaded config dictionary (e.g. from PyYAML). If provided, all
        keys and values are written to the log file immediately after setup.
        Nested dictionaries are indented for readability.

    Returns
    -------
    logging.Logger

    Example
    -------
    In your train.py or ensemble_eval.py:

        import yaml
        from src.utils.logger import get_run_logger

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        logger = get_run_logger(run_dir, __name__, config=config)
        logger.info("Training started.")
    """
    log_file = run_dir / "run.log"
    logger = get_logger(module_name, log_file=log_file)

    # ------------------------------------------------------------------
    # Log the full config at the top of the file so every run is
    # traceable even if the config file is changed between experiments.
    # ------------------------------------------------------------------
    if config is not None:
        logger.info("=" * 60)
        logger.info("RUN CONFIGURATION")
        logger.info("=" * 60)
        _log_config(logger, config)
        logger.info("=" * 60)

    return logger