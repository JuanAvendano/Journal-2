"""
===============================================================
  Dataset Distribution Script
  Source : 02-total-compilation
  Target : DataEnsemble_v2
===============================================================

QUICK-START — only edit the CONFIG block below.

USAGE
-----
  Normal run (distribute images):
      python distribute_dataset.py

  Clean target folders first, then distribute:
      python distribute_dataset.py --clean

  Preview what clean would delete (no files touched):
      python distribute_dataset.py --clean --dry-run

  Only clean, do not distribute:
      python distribute_dataset.py --clean --only-clean
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────

# Absolute (or relative) paths
SOURCE_ROOT = Path(r"D:\JCA\07-Data\01_Concrete\02-total_compilation")
TARGET_ROOT = Path(r"D:\JCA\07-Data\01_Concrete\DataEnsemble_v3")

# Reproducibility
RANDOM_SEED = 42

# ── Split ratios (must sum to 1.0) ───────────────────────────
TRAIN_RATIO = 0.70   # proportion of non-test images → train
VAL_RATIO   = 0.15   # proportion of non-test images → validation
TEST_RATIO  = 0.15   # proportion of total images    → test
# (TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0)

# ── Supported image extensions ────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Per-class configuration ───────────────────────────────────
#
#   max_images   : hard cap on total images taken from source
#                  set to None to use all available images
#
#   weight       : relative sampling weight.
#                  1.0  → baseline
#                  2.0  → this class gets ~2× as many images
#                         as a weight-1 class of the same size
#                  Use this to bias the dataset toward one class
#                  without changing max_images manually.
#
#   Example — "crack expert" setup:
#     crack        weight=3.0, max_images=None  (use everything)
#     everything else weight=1.0
#
CLASS_CONFIG = {
    #  folder name          max_images   weight
    "Crack":        {"max_images": 500,  "weight": 1.0},
    "Efflorescence":{"max_images": 500,  "weight": 1.0},
    "Spalling":     {"max_images": 500,  "weight": 1.0},
    "Undamaged":    {"max_images": 500,  "weight": 1.0},
}

# ── Clean behaviour ──────────────────────────────────────────
#
#   CLEAN_CONFIRM : True  → always ask "are you sure?" before deleting
#                   False → delete without asking (useful in scripts)
#
CLEAN_CONFIRM = True

# ── Folder name mapping ───────────────────────────────────────
#   source folder  →  (train/val subfolder, test subfolder)
FOLDER_MAP = {
    "Crack":         ("crack",         "Crack"),
    "Efflorescence": ("efflorescence", "Efflorescence"),
    "Spalling":      ("spalling",      "Spalling"),
    "Undamaged":     ("undamaged",     "Undamaged"),
}

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def collect_images(folder: Path) -> list[Path]:
    """Return all image files inside a folder (non-recursive top-level only)."""
    images_dir = folder / "images"
    search_dir = images_dir if images_dir.is_dir() else folder
    return [
        f for f in search_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def apply_weight(files: list[Path], weight: float, max_images: int | None) -> list[Path]:
    """
    Apply weight multiplier: conceptually duplicates the pool so that
    weighted sampling draws proportionally more from this class.
    Then cap at max_images.
    """
    # Weight is implemented by repeating the list (rounded), then sampling
    if weight != 1.0:
        repeat = max(1, round(weight))
        files = (files * repeat)[:len(files) * repeat]
    # Shuffle so repeated entries are interleaved, not stacked
    random.shuffle(files)
    # Deduplicate while preserving order (keeps first occurrence)
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    files = unique
    if max_images is not None:
        files = files[:max_images]
    return files


def split_files(files: list[Path], train_r: float, val_r: float, test_r: float):
    """
    Split files into train / val / test.
    test_r is taken first, then train/val split the remainder.
    """
    n = len(files)
    n_test  = round(n * test_r)
    n_train = round((n - n_test) * (train_r / (train_r + val_r)))
    n_val   = n - n_test - n_train

    test  = files[:n_test]
    train = files[n_test: n_test + n_train]
    val   = files[n_test + n_train:]
    return train, val, test


def copy_files(files: list[Path], dest_dir: Path, label: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dest_dir / src.name
        # Avoid silent collisions: append a counter suffix
        if dst.exists():
            stem, suffix = src.stem, src.suffix
            counter = 1
            while dst.exists():
                dst = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(src, dst)


def separator(char="─", width=62):
    print(char * width)

# ──────────────────────────────────────────────────────────────
#  VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_config():
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0 "
            f"(currently {total:.4f})"
        )

# ──────────────────────────────────────────────────────────────
#  CLEAN MODULE
# ──────────────────────────────────────────────────────────────

def get_target_class_dirs() -> list[Path]:
    """Return every class-level leaf folder inside the target tree."""
    dirs = []
    for split_folder in ["01-train", "02-validation", "03-test"]:
        split_path = TARGET_ROOT / split_folder
        if split_path.is_dir():
            dirs.extend([d for d in split_path.iterdir() if d.is_dir()])
    return dirs


def audit_target() -> dict[Path, int]:
    """Return {folder: image_count} for every class folder in the target."""
    result = {}
    for d in get_target_class_dirs():
        count = sum(
            1 for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        result[d] = count
    return result


def print_clean_preview(audit: dict[Path, int], dry_run: bool):
    """Print a formatted table of what will be (or would be) deleted."""
    separator("═")
    tag = "DRY-RUN PREVIEW" if dry_run else "CLEAN TARGET"
    print(f"  {tag} — folders that will be wiped")
    separator("═")

    total_files = 0
    for folder, count in audit.items():
        # Show path relative to TARGET_ROOT for readability
        try:
            rel = folder.relative_to(TARGET_ROOT)
        except ValueError:
            rel = folder
        status = "(empty)" if count == 0 else f"{count} image(s)"
        print(f"  {str(rel):<45}  {status}")
        total_files += count

    separator()
    print(f"  Total images that will be deleted : {total_files}")
    if dry_run:
        print("  [DRY-RUN] No files were deleted.")
    separator("═")


def clean_target(dry_run: bool = False):
    """
    Wipe all images from every class folder inside TARGET_ROOT.
    Folder structure is preserved — only image files are removed.

    dry_run=True : prints what would be deleted, touches nothing.
    """
    audit = audit_target()

    if not audit:
        print("[CLEAN] No class folders found in target — nothing to clean.")
        return

    print_clean_preview(audit, dry_run=dry_run)

    if dry_run:
        return

    # Confirmation prompt (skippable via CLEAN_CONFIRM = False)
    if CLEAN_CONFIRM:
        answer = input("\n  Proceed with deletion? [yes / no] : ").strip().lower()
        if answer not in {"yes", "y"}:
            print("  Aborted. No files deleted.")
            return

    # Delete image files only — folder skeleton stays intact
    deleted_total = 0
    for folder, count in audit.items():
        deleted = 0
        for f in list(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                f.unlink()
                deleted += 1
        deleted_total += deleted
        try:
            rel = folder.relative_to(TARGET_ROOT)
        except ValueError:
            rel = folder
        print(f"  [CLEAN] {str(rel):<45}  deleted {deleted} file(s)")

    separator()
    print(f"  Done. {deleted_total} image(s) removed. Folder structure preserved.")
    separator("═")

# ──────────────────────────────────────────────────────────────
#  ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distribute images from source dataset into train/val/test splits."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe existing images from the target folders before distributing."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what --clean would delete without touching any files."
             " Implies --clean."
    )
    parser.add_argument(
        "--only-clean",
        action="store_true",
        help="Run the clean step only — skip distribution entirely."
    )
    return parser.parse_args()

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    validate_config()
    random.seed(RANDOM_SEED)

    # ── Clean step ─────────────────────────────────────────────
    if args.dry_run or args.clean or args.only_clean:
        dry = args.dry_run
        clean_target(dry_run=dry)
        if args.only_clean or args.dry_run:
            return   # stop here, do not distribute
        print()      # blank line before distribution output

    # ── Distribution step ──────────────────────────────────────
    summary = {}   # class → {train, val, test, available}

    for source_name, cfg in CLASS_CONFIG.items():
        tv_name, test_name = FOLDER_MAP[source_name]
        source_dir = SOURCE_ROOT / source_name

        if not source_dir.is_dir():
            print(f"[WARN] Source folder not found: {source_dir} — skipping.")
            continue

        all_files = collect_images(source_dir)
        available = len(all_files)
        random.shuffle(all_files)

        # Apply weight + cap
        selected = apply_weight(all_files, cfg["weight"], cfg["max_images"])

        # Split
        train_files, val_files, test_files = split_files(
            selected, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        # Copy
        copy_files(train_files, TARGET_ROOT / "01-train"     / tv_name,   "train")
        copy_files(val_files,   TARGET_ROOT / "02-validation" / tv_name,  "val")
        copy_files(test_files,  TARGET_ROOT / "03-test"      / test_name, "test")

        summary[source_name] = {
            "available": available,
            "selected":  len(selected),
            "train":     len(train_files),
            "val":       len(val_files),
            "test":      len(test_files),
            "weight":    cfg["weight"],
            "max":       cfg["max_images"],
        }

    # ── Summary Report ─────────────────────────────────────────
    separator("═")
    print("  DATASET DISTRIBUTION REPORT")
    separator("═")

    total_train = sum(v["train"] for v in summary.values())
    total_val   = sum(v["val"]   for v in summary.values())
    total_test  = sum(v["test"]  for v in summary.values())
    total_tv    = total_train + total_val
    total_all   = total_tv + total_test

    actual_train_pct = (total_train / total_tv  * 100) if total_tv  else 0
    actual_val_pct   = (total_val   / total_tv  * 100) if total_tv  else 0
    actual_test_pct  = (total_test  / total_all * 100) if total_all else 0

    print(f"\n  Seed : {RANDOM_SEED}")
    print(f"  Split (configured) : "
          f"Train {TRAIN_RATIO*100:.0f}% / "
          f"Val {VAL_RATIO*100:.0f}% / "
          f"Test {TEST_RATIO*100:.0f}%\n")

    separator()
    print(f"  {'CLASS':<18} {'AVAIL':>6} {'USED':>6} {'TRAIN':>7} {'VAL':>7} {'TEST':>7}  WEIGHT")
    separator()

    for cls, v in summary.items():
        print(
            f"  {cls:<18} "
            f"{v['available']:>6} "
            f"{v['selected']:>6} "
            f"{v['train']:>7} "
            f"{v['val']:>7} "
            f"{v['test']:>7}  "
            f"×{v['weight']:.1f}"
        )

    separator()

    # ── Train set class balance ────────────────────────────────
    print(f"\n  TRAIN SET  ({total_train} images)")
    separator("-", 40)
    for cls, v in summary.items():
        pct = (v["train"] / total_train * 100) if total_train else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:<18} {v['train']:>5}  {pct:5.1f}%  {bar}")

    # ── Validation set class balance ───────────────────────────
    print(f"\n  VALIDATION SET  ({total_val} images)")
    separator("-", 40)
    for cls, v in summary.items():
        pct = (v["val"] / total_val * 100) if total_val else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:<18} {v['val']:>5}  {pct:5.1f}%  {bar}")

    # ── Test set class balance ─────────────────────────────────
    print(f"\n  TEST SET  ({total_test} images)")
    separator("-", 40)
    for cls, v in summary.items():
        pct = (v["test"] / total_test * 100) if total_test else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:<18} {v['test']:>5}  {pct:5.1f}%  {bar}")

    # ── Grand totals ───────────────────────────────────────────
    separator()
    print(f"\n  TOTALS")
    print(f"  {'Train + Val (model dataset)':<30} {total_tv:>5}  "
          f"({actual_train_pct:.1f}% train / {actual_val_pct:.1f}% val)")
    print(f"  {'Test (held-out)':<30} {total_test:>5}  "
          f"({actual_test_pct:.1f}% of total)")
    print(f"  {'Grand total':<30} {total_all:>5}")
    separator("═")


if __name__ == "__main__":
    main()