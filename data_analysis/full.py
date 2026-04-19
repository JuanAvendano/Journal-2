"""
Created on Monday Mar 23 2026
Advanced dataset analysis for image segmentation JSON labels.

Outputs:
- CSV files
- JSON files
- TXT summary
- plots

Features:
- Global class summary
- Split-wise class summary
- Image-level summary
- Object-level summary
- Per-class image lists
- Co-occurrence matrix
- Object size distributions
- Images with no objects
- Images with many tiny objects
- Optional exact polygon-union labeled area if polygons exist

Expected minimum JSON structure:
{
    "info": {
        "name": "image.jpg",
        "width": 1600,
        "height": 1200
    },
    "objects": [
        {
            "category": "Crack",
            "area": 1983.84
        }
    ]
}

Optional polygon support:
Each object may also include one of:
- "segmentation"
- "polygon"
- "points"
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional import for exact polygon rasterization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Configuration
# =============================================================================

DATASET_ROOT = Path(r"D:/JCA/07-Data/data_analysis_try/")
DATASET_NAME = DATASET_ROOT.name
OUTPUT_DIR = DATASET_ROOT / f"_analysis_output_{DATASET_NAME}"

SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

BACKGROUND_CLASS_NAMES = {"background", "undamaged", "unlabeled"}

# If True, remove these classes from class-based statistics and co-occurrence.
EXCLUDE_BACKGROUND_CLASSES_FROM_CLASS_STATS = False

# If True, compute labeled area from exact polygon union when polygons exist.
# Requires OpenCV. Falls back to summed object areas otherwise.
USE_EXACT_POLYGON_UNION_IF_AVAILABLE = True

SAVE_CSV = True
SAVE_JSON = True
SAVE_TXT = True
SHOW_PLOTS = True

# Threshold used for “tiny object” report
TINY_OBJECT_AREA_THRESHOLD = 500.0


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: Any) -> str:
    return str(name).strip()


def detect_split_from_path(path: Path) -> str:
    for part in path.parts:
        p = part.lower()
        if p in SPLIT_NAMES:
            return SPLIT_NAMES[p]
    return "unknown"


def find_json_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def safe_load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read {path}: {e}")
        return None


def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def classify_object_size(area: float) -> str:
    if pd.isna(area):
        return "unknown"
    if area < 32**2:
        return "small"
    elif area < 96**2:
        return "medium"
    else:
        return "large"


def percentile_safe(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, q))


def save_dataframe_as_json(df: pd.DataFrame, path: Path) -> None:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# =============================================================================
# Polygon parsing / rasterization
# =============================================================================

def try_extract_polygon_points(obj: dict) -> list[np.ndarray]:
    """
    Attempts to extract polygons from various possible formats.

    Returns a list of polygons, each polygon as Nx2 array.
    If none found, returns [].
    """
    polygons = []

    # Case 1: segmentation = [[x1,y1,x2,y2,...], [...]]
    seg = obj.get("segmentation")
    if isinstance(seg, list) and len(seg) > 0:
        if all(isinstance(item, list) for item in seg):
            for poly in seg:
                if len(poly) >= 6 and len(poly) % 2 == 0:
                    arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    polygons.append(arr)

    # Case 2: polygon = [[x,y], [x,y], ...]
    poly = obj.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        try:
            arr = np.array(poly, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                polygons.append(arr)
        except Exception:
            pass

    # Case 3: points = [[x,y], [x,y], ...]
    pts = obj.get("points")
    if isinstance(pts, list) and len(pts) >= 3:
        try:
            arr = np.array(pts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                polygons.append(arr)
        except Exception:
            pass

    return polygons


def polygon_union_area(image_h: int, image_w: int, polygons: list[np.ndarray]) -> float:
    """
    Rasterize polygons into a binary mask and compute union area in pixels.
    Requires cv2.
    """
    if not HAS_CV2 or not polygons:
        return np.nan

    mask = np.zeros((int(image_h), int(image_w)), dtype=np.uint8)

    for poly in polygons:
        poly_i = np.round(poly).astype(np.int32)
        poly_i[:, 0] = np.clip(poly_i[:, 0], 0, image_w - 1)
        poly_i[:, 1] = np.clip(poly_i[:, 1], 0, image_h - 1)
        cv2.fillPoly(mask, [poly_i], 1)

    return float(mask.sum())


# =============================================================================
# Parsing
# =============================================================================

def parse_annotation(json_path: Path) -> dict | None:
    data = safe_load_json(json_path)
    if data is None:
        return None

    info = data.get("info", {})
    objects = data.get("objects", [])

    image_name = info.get("name", json_path.with_suffix(".jpg").name)
    width = info.get("width", np.nan)
    height = info.get("height", np.nan)

    try:
        width = int(width)
    except Exception:
        width = np.nan

    try:
        height = int(height)
    except Exception:
        height = np.nan

    parsed_objects = []
    all_polygons = []

    for obj in objects:
        category = normalize_class_name(obj.get("category", "Unknown"))

        area = obj.get("area", np.nan)
        try:
            area = float(area)
        except Exception:
            area = np.nan

        polygons = try_extract_polygon_points(obj)
        if polygons:
            all_polygons.extend(polygons)

        parsed_objects.append({
            "category": category,
            "area": area,
            "polygons": polygons,
            "size_bin": classify_object_size(area),
        })

    image_area = (width * height) if pd.notna(width) and pd.notna(height) else np.nan

    exact_union_area = np.nan
    if (
        USE_EXACT_POLYGON_UNION_IF_AVAILABLE
        and HAS_CV2
        and pd.notna(width)
        and pd.notna(height)
        and len(all_polygons) > 0
    ):
        exact_union_area = polygon_union_area(height, width, all_polygons)

    return {
        "json_path": str(json_path),
        "image_name": image_name,
        "split": detect_split_from_path(json_path),
        "width": width,
        "height": height,
        "image_area": image_area,
        "objects": parsed_objects,
        "exact_union_labeled_area_px2": exact_union_area,
    }


# =============================================================================
# Plotting
# =============================================================================

def save_or_show(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def make_plots_v2(class_summary_df: pd.DataFrame,
                  split_class_summary_df: pd.DataFrame,
                  image_df: pd.DataFrame,
                  object_df: pd.DataFrame,
                  cooc_matrix: pd.DataFrame,
                  output_dir: Path,
                  dataset_name: str) -> None:
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)

    prefix = sanitize_filename(dataset_name)

    # 1. Images containing each class
    if not class_summary_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(class_summary_df["class"], class_summary_df["images_with_at_least_one_object"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Images")
        plt.title(f"Images containing each class\nDataset: {dataset_name}")
        save_or_show(plots_dir / f"{prefix}__images_per_class.png")

    # 2. Total objects per class
    if not class_summary_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(class_summary_df["class"], class_summary_df["objects_total"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Objects")
        plt.title(f"Total objects per class\nDataset: {dataset_name}")
        save_or_show(plots_dir / f"{prefix}__objects_per_class.png")

    # 3. Average object area per class
    if not class_summary_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(class_summary_df["class"], class_summary_df["avg_area_per_object_px2"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average area (px²)")
        plt.title(f"Average object area per class\nDataset: {dataset_name}")
        save_or_show(plots_dir / f"{prefix}__avg_area_per_object_per_class.png")

    # 4. Co-occurrence heatmap
    if not cooc_matrix.empty:
        plt.figure(figsize=(8, 6))
        plt.imshow(cooc_matrix.values, aspect="auto")
        plt.colorbar(label="Number of images")
        plt.xticks(range(len(cooc_matrix.columns)), cooc_matrix.columns, rotation=45, ha="right")
        plt.yticks(range(len(cooc_matrix.index)), cooc_matrix.index)
        plt.title(f"Class co-occurrence matrix\nDataset: {dataset_name}")
        save_or_show(plots_dir / f"{prefix}__cooccurrence_heatmap.png")

    # 5. Unlabeled percentage histogram
    if not image_df.empty:
        vals = image_df["unlabeled_pct"].dropna()
        if len(vals) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=30)
            plt.xlabel("Unlabeled area (%)")
            plt.ylabel("Images")
            plt.title(f"Distribution of unlabeled area percentage\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__unlabeled_pct_histogram.png")

    # 6. Number of classes per image
    if not image_df.empty:
        vals = image_df["num_present_classes"].dropna().astype(int)
        if len(vals) > 0:
            bins = np.arange(vals.min(), vals.max() + 2) - 0.5
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=bins)
            plt.xlabel("Number of classes present")
            plt.ylabel("Images")
            plt.title(f"Class richness per image\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__classes_per_image_histogram.png")

    # 7. Number of objects per image
    if not image_df.empty:
        vals = image_df["num_objects_total"].dropna()
        if len(vals) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=30)
            plt.xlabel("Number of objects in image")
            plt.ylabel("Images")
            plt.title(f"Object count per image\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__objects_per_image_histogram.png")

    # 8. Split-wise class prevalence
    if not split_class_summary_df.empty:
        pivot = split_class_summary_df.pivot(index="class", columns="split", values="images_with_class").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel("Images with class")
        ax.set_title(f"Class prevalence by split\nDataset: {dataset_name}")
        plt.xticks(rotation=45, ha="right")
        save_or_show(plots_dir / f"{prefix}__class_prevalence_by_split.png")

    # 9. Object area histograms per class
    if not object_df.empty:
        classes = sorted(object_df["class"].dropna().unique())
        for cls in classes:
            vals = object_df.loc[object_df["class"] == cls, "area_px2"].dropna()
            if len(vals) == 0:
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=30)
            plt.xlabel("Object area (px²)")
            plt.ylabel("Objects")
            plt.title(f"Object area distribution - {cls}\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__hist_object_area__{sanitize_filename(cls)}.png")

    # 10. Log-area histograms per class
    if not object_df.empty:
        classes = sorted(object_df["class"].dropna().unique())
        for cls in classes:
            vals = object_df.loc[object_df["class"] == cls, "area_px2"].dropna()
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(np.log10(vals), bins=30)
            plt.xlabel("log10(Object area px²)")
            plt.ylabel("Objects")
            plt.title(f"Log object area distribution - {cls}\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__hist_log_object_area__{sanitize_filename(cls)}.png")

    # 11. Boxplot of object area by class
    if not object_df.empty:
        classes = sorted(object_df["class"].dropna().unique())
        data = [
            object_df.loc[(object_df["class"] == cls) & (object_df["area_px2"] > 0), "area_px2"].values
            for cls in classes
        ]
        if any(len(d) > 0 for d in data):
            plt.figure(figsize=(12, 6))
            plt.boxplot(data, tick_labels=classes, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Object area (px²)")
            plt.title(f"Object area by class\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__boxplot_object_area_by_class.png")

    # 12. Boxplot of log object area by class
    if not object_df.empty:
        classes = sorted(object_df["class"].dropna().unique())
        data = []
        valid_labels = []
        for cls in classes:
            vals = object_df.loc[(object_df["class"] == cls) & (object_df["area_px2"] > 0), "area_px2"].values
            if len(vals) > 0:
                data.append(np.log10(vals))
                valid_labels.append(cls)
        if data:
            plt.figure(figsize=(12, 6))
            plt.boxplot(data, tick_labels=valid_labels, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("log10(Object area px²)")
            plt.title(f"Log object area by class\nDataset: {dataset_name}")
            save_or_show(plots_dir / f"{prefix}__boxplot_log_object_area_by_class.png")


# =============================================================================
# Text summary
# =============================================================================

def print_text_summary(results: dict) -> None:
    print("\n==================== DATASET SUMMARY ====================")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Dataset path: {DATASET_ROOT}")
    print(f"Total images: {len(results['image_df'])}")
    print(f"Total objects: {len(results['object_df'])}")
    print(f"Classes: {results['all_classes']}")

    print("\n-------------------- Split summary --------------------")
    print(results["split_summary_df"].to_string(index=False))

    print("\n-------------------- Global class summary --------------------")
    print(results["class_summary_df"].to_string(index=False))

    print("\n-------------------- Images with no objects --------------------")
    print(f"Count: {len(results['empty_images_df'])}")

    if not results["many_small_objects_per_image_df"].empty:
        print("\n-------------------- Images with many tiny objects --------------------")
        print(results["many_small_objects_per_image_df"].head(10).to_string(index=False))


def save_text_summary(results: dict, output_dir: Path, dataset_name: str, dataset_root: Path) -> None:
    prefix = sanitize_filename(dataset_name)
    txt_path = output_dir / f"{prefix}__summary.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DATASET ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset name : {dataset_name}\n")
        f.write(f"Dataset path : {dataset_root}\n")
        f.write(f"Output path  : {output_dir}\n")
        f.write("\n")

        f.write("GENERAL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total images  : {len(results['image_df'])}\n")
        f.write(f"Total objects : {len(results['object_df'])}\n")
        f.write(f"Classes       : {results['all_classes']}\n")
        f.write("\n")

        f.write("SPLIT SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(results["split_summary_df"].to_string(index=False))
        f.write("\n\n")

        f.write("GLOBAL CLASS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(results["class_summary_df"].to_string(index=False))
        f.write("\n\n")

        f.write("IMAGES WITH NO OBJECTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(results['empty_images_df'])}\n")
        if not results["empty_images_df"].empty:
            f.write(results["empty_images_df"][["image_name", "split"]].to_string(index=False))
            f.write("\n\n")

        if not results["many_small_objects_per_image_df"].empty:
            f.write("TOP IMAGES WITH MANY TINY OBJECTS\n")
            f.write("-" * 80 + "\n")
            f.write(results["many_small_objects_per_image_df"].head(20).to_string(index=False))
            f.write("\n")


# =============================================================================
# Core analysis
# =============================================================================

def analyze_dataset(dataset_root: Path, output_dir: Path) -> dict:
    ensure_dir(output_dir)

    json_files = find_json_files(dataset_root)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {dataset_root}")

    annotations = []
    for jf in json_files:
        parsed = parse_annotation(jf)
        if parsed is not None:
            annotations.append(parsed)

    if not annotations:
        raise RuntimeError("No valid annotations were parsed.")

    all_classes = sorted({
        obj["category"]
        for ann in annotations
        for obj in ann["objects"]
    })

    classes_for_stats = all_classes.copy()
    if EXCLUDE_BACKGROUND_CLASSES_FROM_CLASS_STATS:
        classes_for_stats = [
            c for c in all_classes
            if c.lower() not in BACKGROUND_CLASS_NAMES
        ]

    # -------------------------------------------------------------------------
    # Object-level table
    # -------------------------------------------------------------------------
    object_rows = []
    for ann in annotations:
        for idx, obj in enumerate(ann["objects"]):
            object_rows.append({
                "image_name": ann["image_name"],
                "split": ann["split"],
                "object_index": idx,
                "class": obj["category"],
                "area_px2": obj["area"],
                "size_bin": obj["size_bin"],
                "has_polygon": len(obj["polygons"]) > 0,
                "json_path": ann["json_path"],
            })

    object_df = pd.DataFrame(object_rows)
    if object_df.empty:
        object_df = pd.DataFrame(columns=[
            "image_name", "split", "object_index", "class",
            "area_px2", "size_bin", "has_polygon", "json_path"
        ])

    object_df.insert(0, "dataset_name", DATASET_NAME)
    object_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Image-level table
    # -------------------------------------------------------------------------
    image_rows = []
    image_to_present_classes = {}
    image_to_class_counts = {}
    image_to_class_areas = {}

    for ann in annotations:
        counts = Counter()
        areas = Counter()

        for obj in ann["objects"]:
            counts[obj["category"]] += 1
            if pd.notna(obj["area"]):
                areas[obj["category"]] += float(obj["area"])

        present_classes = sorted([cls for cls, cnt in counts.items() if cnt > 0])

        summed_labeled_area = float(sum(areas.values()))
        exact_union = ann["exact_union_labeled_area_px2"]

        if pd.notna(exact_union):
            labeled_area = exact_union
            labeled_area_method = "polygon_union"
        else:
            labeled_area = summed_labeled_area
            labeled_area_method = "sum_of_object_areas"

        image_area = ann["image_area"]
        unlabeled_area = np.nan
        unlabeled_pct = np.nan
        labeled_pct = np.nan

        if pd.notna(image_area) and image_area > 0 and pd.notna(labeled_area):
            unlabeled_area = max(0.0, image_area - labeled_area)
            unlabeled_pct = 100.0 * unlabeled_area / image_area
            labeled_pct = 100.0 * labeled_area / image_area

        row = {
            "image_name": ann["image_name"],
            "split": ann["split"],
            "width_px": ann["width"],
            "height_px": ann["height"],
            "image_area_px2": image_area,
            "labeled_area_px2": labeled_area,
            "labeled_area_method": labeled_area_method,
            "unlabeled_area_px2": unlabeled_area,
            "labeled_pct": labeled_pct,
            "unlabeled_pct": unlabeled_pct,
            "num_present_classes": len(present_classes),
            "present_classes": ", ".join(present_classes),
            "num_objects_total": int(sum(counts.values())),
            "json_path": ann["json_path"],
        }

        for cls in all_classes:
            row[f"count__{cls}"] = int(counts.get(cls, 0))
            row[f"area__{cls}"] = float(areas.get(cls, 0.0))

        image_rows.append(row)
        image_to_present_classes[ann["image_name"]] = set(present_classes)
        image_to_class_counts[ann["image_name"]] = counts
        image_to_class_areas[ann["image_name"]] = areas

    image_df = pd.DataFrame(image_rows).sort_values(["split", "image_name"]).reset_index(drop=True)
    image_df.insert(0, "dataset_name", DATASET_NAME)
    image_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Global class summary
    # -------------------------------------------------------------------------
    images_with_class = defaultdict(list)
    images_only_class = defaultdict(list)
    global_class_rows = []

    total_images = len(image_df)

    for cls in classes_for_stats:
        imgs_with = []
        imgs_only = []
        obj_count_total = 0
        class_area_total = 0.0

        for _, row in image_df.iterrows():
            img = row["image_name"]
            cnt = int(row[f"count__{cls}"])
            area = float(row[f"area__{cls}"])
            present = image_to_present_classes[img]

            if cnt > 0:
                imgs_with.append(img)
                obj_count_total += cnt
                class_area_total += area

            if present == {cls}:
                imgs_only.append(img)

        images_with_class[cls] = sorted(imgs_with)
        images_only_class[cls] = sorted(imgs_only)

        n_img_with = len(imgs_with)

        cls_object_series = object_df.loc[object_df["class"] == cls, "area_px2"] if not object_df.empty else pd.Series(dtype=float)

        global_class_rows.append({
            "class": cls,
            "images_with_at_least_one_object": n_img_with,
            "objects_total": obj_count_total,
            "avg_objects_per_image_with_class": obj_count_total / n_img_with if n_img_with > 0 else np.nan,
            "avg_class_area_per_image_with_class_px2": class_area_total / n_img_with if n_img_with > 0 else np.nan,
            "avg_area_per_object_px2": class_area_total / obj_count_total if obj_count_total > 0 else np.nan,
            "median_area_per_object_px2": cls_object_series.median() if len(cls_object_series) > 0 else np.nan,
            "p10_area_per_object_px2": percentile_safe(cls_object_series, 10) if len(cls_object_series) > 0 else np.nan,
            "p90_area_per_object_px2": percentile_safe(cls_object_series, 90) if len(cls_object_series) > 0 else np.nan,
            "images_with_only_this_class": len(imgs_only),
            "prevalence_pct_of_images": 100.0 * n_img_with / total_images if total_images > 0 else np.nan,
        })

    class_summary_df = pd.DataFrame(global_class_rows).sort_values("class").reset_index(drop=True)
    class_summary_df.insert(0, "dataset_name", DATASET_NAME)
    class_summary_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Split-wise class summary
    # -------------------------------------------------------------------------
    split_class_rows = []

    for split_name, split_subdf in image_df.groupby("split"):
        n_split_images = len(split_subdf)

        for cls in classes_for_stats:
            imgs_with = split_subdf[split_subdf[f"count__{cls}"] > 0]
            obj_total = imgs_with[f"count__{cls}"].sum()
            area_total = imgs_with[f"area__{cls}"].sum()

            split_class_rows.append({
                "split": split_name,
                "class": cls,
                "images_with_class": int(len(imgs_with)),
                "objects_total": int(obj_total),
                "avg_objects_per_image_with_class": obj_total / len(imgs_with) if len(imgs_with) > 0 else np.nan,
                "avg_class_area_per_image_with_class_px2": area_total / len(imgs_with) if len(imgs_with) > 0 else np.nan,
                "prevalence_pct_within_split": 100.0 * len(imgs_with) / n_split_images if n_split_images > 0 else np.nan,
            })

    split_class_summary_df = pd.DataFrame(split_class_rows).sort_values(["split", "class"]).reset_index(drop=True)
    split_class_summary_df.insert(0, "dataset_name", DATASET_NAME)
    split_class_summary_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Split summary
    # -------------------------------------------------------------------------
    split_rows = []
    for split_name, subdf in image_df.groupby("split"):
        split_rows.append({
            "split": split_name,
            "num_images": len(subdf),
            "mean_width_px": subdf["width_px"].mean(),
            "mean_height_px": subdf["height_px"].mean(),
            "mean_num_objects_total": subdf["num_objects_total"].mean(),
            "mean_num_present_classes": subdf["num_present_classes"].mean(),
            "mean_labeled_pct": subdf["labeled_pct"].mean(),
            "mean_unlabeled_pct": subdf["unlabeled_pct"].mean(),
            "median_unlabeled_pct": subdf["unlabeled_pct"].median(),
        })
    split_summary_df = pd.DataFrame(split_rows).sort_values("split").reset_index(drop=True)
    split_summary_df.insert(0, "dataset_name", DATASET_NAME)
    split_summary_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Co-occurrence matrix
    # -------------------------------------------------------------------------
    cooc_classes = classes_for_stats.copy()
    cooc_matrix = pd.DataFrame(0, index=cooc_classes, columns=cooc_classes, dtype=int)

    for _, row in image_df.iterrows():
        present = [cls for cls in cooc_classes if row[f"count__{cls}"] > 0]
        for a, b in combinations_with_replacement(sorted(present), 2):
            cooc_matrix.loc[a, b] += 1
            if a != b:
                cooc_matrix.loc[b, a] += 1

    # -------------------------------------------------------------------------
    # Object size distribution by class
    # -------------------------------------------------------------------------
    if not object_df.empty:
        size_dist_df = (
            object_df.groupby(["class", "size_bin"])
            .size()
            .reset_index(name="object_count")
            .sort_values(["class", "size_bin"])
            .reset_index(drop=True)
        )
    else:
        size_dist_df = pd.DataFrame(columns=["class", "size_bin", "object_count"])

    size_dist_df.insert(0, "dataset_name", DATASET_NAME)
    size_dist_df.insert(1, "dataset_root", str(DATASET_ROOT))

    # -------------------------------------------------------------------------
    # Special reports
    # -------------------------------------------------------------------------
    empty_images_df = image_df[image_df["num_objects_total"] == 0].copy()

    if not object_df.empty:
        tiny_objects_df = object_df[object_df["area_px2"] < TINY_OBJECT_AREA_THRESHOLD].copy()
    else:
        tiny_objects_df = pd.DataFrame(columns=object_df.columns)

    if not tiny_objects_df.empty:
        many_small_objects_per_image_df = (
            tiny_objects_df.groupby(["image_name", "split"])
            .size()
            .reset_index(name="num_tiny_objects")
            .sort_values("num_tiny_objects", ascending=False)
            .reset_index(drop=True)
        )
        many_small_objects_per_image_df.insert(0, "dataset_name", DATASET_NAME)
        many_small_objects_per_image_df.insert(1, "dataset_root", str(DATASET_ROOT))
    else:
        many_small_objects_per_image_df = pd.DataFrame(
            columns=["dataset_name", "dataset_root", "image_name", "split", "num_tiny_objects"]
        )

    # -------------------------------------------------------------------------
    # Per-class image lists
    # -------------------------------------------------------------------------
    class_lists_dir = output_dir / "class_image_lists"
    ensure_dir(class_lists_dir)
    prefix = sanitize_filename(DATASET_NAME)

    for cls in all_classes:
        safe_cls = sanitize_filename(cls)

        df_with = pd.DataFrame({
            "dataset_name": [DATASET_NAME] * len(images_with_class.get(cls, [])),
            "dataset_root": [str(DATASET_ROOT)] * len(images_with_class.get(cls, [])),
            "class": [cls] * len(images_with_class.get(cls, [])),
            "image_name": sorted(images_with_class.get(cls, []))
        })
        df_with.to_csv(
            class_lists_dir / f"{prefix}__images_with__{safe_cls}.csv",
            index=False
        )

        df_only = pd.DataFrame({
            "dataset_name": [DATASET_NAME] * len(images_only_class.get(cls, [])),
            "dataset_root": [str(DATASET_ROOT)] * len(images_only_class.get(cls, [])),
            "class": [cls] * len(images_only_class.get(cls, [])),
            "image_name": sorted(images_only_class.get(cls, []))
        })
        df_only.to_csv(
            class_lists_dir / f"{prefix}__images_only__{safe_cls}.csv",
            index=False
        )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------
    if SAVE_CSV:
        class_summary_df.to_csv(output_dir / f"{prefix}__class_summary.csv", index=False)
        split_class_summary_df.to_csv(output_dir / f"{prefix}__split_class_summary.csv", index=False)
        split_summary_df.to_csv(output_dir / f"{prefix}__split_summary.csv", index=False)
        image_df.to_csv(output_dir / f"{prefix}__image_summary.csv", index=False)
        object_df.to_csv(output_dir / f"{prefix}__object_summary.csv", index=False)
        size_dist_df.to_csv(output_dir / f"{prefix}__object_size_distribution.csv", index=False)
        empty_images_df.to_csv(output_dir / f"{prefix}__images_with_no_objects.csv", index=False)
        tiny_objects_df.to_csv(output_dir / f"{prefix}__tiny_objects.csv", index=False)
        many_small_objects_per_image_df.to_csv(output_dir / f"{prefix}__images_with_many_tiny_objects.csv", index=False)

        cooc_matrix_to_save = cooc_matrix.copy()
        cooc_matrix_to_save.insert(0, "class", cooc_matrix_to_save.index)
        cooc_matrix_to_save.to_csv(output_dir / f"{prefix}__class_cooccurrence_matrix.csv", index=False)

    if SAVE_JSON:
        save_dataframe_as_json(class_summary_df, output_dir / f"{prefix}__class_summary.json")
        save_dataframe_as_json(split_class_summary_df, output_dir / f"{prefix}__split_class_summary.json")
        save_dataframe_as_json(split_summary_df, output_dir / f"{prefix}__split_summary.json")
        save_dataframe_as_json(image_df, output_dir / f"{prefix}__image_summary.json")
        save_dataframe_as_json(object_df, output_dir / f"{prefix}__object_summary.json")
        save_dataframe_as_json(size_dist_df, output_dir / f"{prefix}__object_size_distribution.json")
        save_dataframe_as_json(empty_images_df, output_dir / f"{prefix}__images_with_no_objects.json")
        save_dataframe_as_json(tiny_objects_df, output_dir / f"{prefix}__tiny_objects.json")
        save_dataframe_as_json(
            many_small_objects_per_image_df,
            output_dir / f"{prefix}__images_with_many_tiny_objects.json"
        )

        with open(output_dir / f"{prefix}__class_cooccurrence_matrix.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": DATASET_NAME,
                    "dataset_root": str(DATASET_ROOT),
                    "matrix": cooc_matrix.to_dict()
                },
                f,
                indent=2,
                ensure_ascii=False
            )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    make_plots_v2(
        class_summary_df=class_summary_df,
        split_class_summary_df=split_class_summary_df,
        image_df=image_df,
        object_df=object_df,
        cooc_matrix=cooc_matrix,
        output_dir=output_dir,
        dataset_name=DATASET_NAME
    )

    return {
        "annotations": annotations,
        "all_classes": all_classes,
        "class_summary_df": class_summary_df,
        "split_class_summary_df": split_class_summary_df,
        "split_summary_df": split_summary_df,
        "image_df": image_df,
        "object_df": object_df,
        "cooc_matrix": cooc_matrix,
        "size_dist_df": size_dist_df,
        "empty_images_df": empty_images_df,
        "tiny_objects_df": tiny_objects_df,
        "many_small_objects_per_image_df": many_small_objects_per_image_df,
        "images_with_class": dict(images_with_class),
        "images_only_class": dict(images_only_class),
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    results = analyze_dataset(DATASET_ROOT, OUTPUT_DIR)
    print_text_summary(results)

    if SAVE_TXT:
        save_text_summary(results, OUTPUT_DIR, DATASET_NAME, DATASET_ROOT)

    print(f"\nSaved analysis to:\n{OUTPUT_DIR}")