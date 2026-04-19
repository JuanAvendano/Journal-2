"""
Created on Monday Mar 23 2026
analyze_dataset.py
==================
Dataset-level analysis for ISAT-format segmentation label files.

Folder structure expected:
    dataset_root/
        train/   *.json
        val/     *.json   (or 'valid', 'validation')
        test/    *.json   (optional)

Usage:
    python analyze_dataset.py --root /path/to/dataset --name "name of the dataset" --out ./dataset_report
    Example:
    For: D:\JCA\07-Data\data_analysis_try
    python analyze_dataset.py --root D:\JCA\07-Data\data_analysis_try\ --out ./dataset_report_dacl10k
    python analyze_dataset.py --root D:\JCA\07-Data\01_Concrete\dacl10k\labels\  --name "dacl10k" --out ./dataset_report_dacl10k
"""

import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset identity
# ═══════════════════════════════════════════════════════════════════════════

def get_dataset_info(root: Path) -> tuple[str, str]:
    """
    Return (dataset_name, dataset_full_path).
    If root points to a split folder (train/val/test), the name comes from
    the parent so we always get the actual dataset name.
    """
    resolved = root.resolve()
    SPLIT_NAMES = {"train","training","val","valid","validation","test","testing"}
    if resolved.name.lower() in SPLIT_NAMES:
        name = resolved.parent.name
        path = str(resolved.parent)
    else:
        name = resolved.name
        path = str(resolved)
    return name, path



def write_csv_with_header(df: pd.DataFrame, filepath: Path,
                          ds_name: str, ds_path: str, description: str):
    """Write CSV with a metadata comment block at the top."""

    header = [
        f"# Dataset   : {ds_name}",
        f"# Full path : {ds_path}",
        f"# Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Content   : {description}",
        "#",
    ]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False)


def save_fig(fig, filepath: Path, ds_name: str, ds_path: str):
    """Stamp a footer with dataset identity then save."""
    fig.text(
        0.5, 0.005,
        f"Dataset: {ds_name}   |   {ds_path}",
        ha="center", va="bottom", fontsize=7,
        color="#555555", style="italic",
    )
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def colour_for(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════

def find_splits(root: Path) -> dict[str, Path]:
    SPLIT_ALIASES = {
        "train": ["train", "training"],
        "val":   ["val", "valid", "validation"],
        "test":  ["test", "testing"],
    }
    # Case A: named sub-folders
    found = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            p = root / alias
            if p.is_dir():
                found[canonical] = p
                break
    if found:
        return found
    # Case B: root itself has JSONs
    if any(root.glob("*.json")):
        folder_name = root.name.lower()
        canonical = "unknown"
        for can, aliases in SPLIT_ALIASES.items():
            if any(folder_name.startswith(a) for a in aliases):
                canonical = can
                break
        print(f"  [auto-detect] Treating '{root.name}' as split='{canonical}'.")
        return {canonical: root}
    # Case C: user passed a split folder — go up one level
    parent = root.parent
    found_parent = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            p = parent / alias
            if p.is_dir():
                found_parent[canonical] = p
                break
    if found_parent:
        print(f"  [auto-detect] Using parent '{parent.name}' as dataset root.")
        return found_parent
    return {}


def load_jsons(root: Path) -> list[dict]:
    splits = find_splits(root)
    if not splits:
        raise FileNotFoundError(
            f"No JSON files or split sub-folders found under:\n  {root}\n"
            f"Point --root at the folder that CONTAINS train/ val/ etc."
        )
    records = []
    for split_name, folder in splits.items():
        jsons = sorted(folder.glob("*.json"))
        print(f"  [{split_name}] {len(jsons)} files  →  {folder}")
        for jf in jsons:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_split"] = split_name
            data["_stem"]  = jf.stem
            data["_path"]  = str(jf)
            records.append(data)
    if not records:
        raise FileNotFoundError(f"No JSON files found under {root}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Data-frame builders
# ═══════════════════════════════════════════════════════════════════════════

def build_flat_df(records):
    rows = []
    for rec in records:
        info = rec.get("info", {})
        img_area = info.get("width", 0) * info.get("height", 0)
        for obj in rec.get("objects", []):
            rows.append({
                "image":    rec["_stem"],
                "split":    rec["_split"],
                "img_w":    info.get("width",  0),
                "img_h":    info.get("height", 0),
                "img_area": img_area,
                "category": obj.get("category", "Unknown"),
                "obj_area": obj.get("area", 0.0),
            })
    return pd.DataFrame(rows)


def build_image_df(records):
    rows = []
    for rec in records:
        info = rec.get("info", {})
        rows.append({
            "image":      rec["_stem"],
            "split":      rec["_split"],
            "img_w":      info.get("width",  0),
            "img_h":      info.get("height", 0),
            "img_area":   info.get("width", 0) * info.get("height", 0),
            "n_objects":  len(rec.get("objects", [])),
            "categories": sorted({o.get("category","Unknown")
                                   for o in rec.get("objects", [])}),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════════════════

def class_summary_table(flat, image_df):
    all_classes = sorted(flat["category"].unique())
    rows = []
    for cls in all_classes:
        sub    = flat[flat["category"] == cls]
        imgs_w = set(sub["image"])
        imgs_only = set(image_df[
            image_df["categories"].apply(lambda c: set(c) == {cls})
        ]["image"])
        n_imgs = len(imgs_w)
        rows.append({
            "Class":                     cls,
            "Images w/ class":           n_imgs,
            "% images w/ class":         round(100 * n_imgs / len(image_df), 1),
            "Total objects":             len(sub),
            "Avg objects / image":       round(sub.groupby("image").size().mean(), 2) if n_imgs else 0,
            "Avg area / image (px²)":    round(sub.groupby("image")["obj_area"].sum().mean(), 1) if n_imgs else 0,
            "Avg area % / image":        round(
                (sub.groupby("image")["obj_area"].sum() /
                 sub.groupby("image")["img_area"].first() * 100).mean(), 2
            ) if n_imgs else 0,
            "Images w/ ONLY this class": len(imgs_only),
        })
    return pd.DataFrame(rows).sort_values("Total objects", ascending=False)


def class_image_lists(flat, image_df):
    result = {}
    for cls in sorted(flat["category"].unique()):
        result[cls] = {
            "all":  sorted(flat[flat["category"] == cls]["image"].unique()),
            "only": sorted(image_df[
                image_df["categories"].apply(lambda c: set(c) == {cls})
            ]["image"]),
        }
    return result


def cooccurrence_matrix(image_df):
    classes = sorted({c for cats in image_df["categories"] for c in cats})
    mat = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for cats in image_df["categories"]:
        cat_set = set(cats)
        for ci in cat_set:
            for cj in cat_set:
                mat.loc[ci, cj] += 1
    return mat


# ═══════════════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_class_counts(summary, out, ds_name, ds_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Object & image counts per class  —  {ds_name}", fontsize=12, fontweight="bold")
    classes = summary["Class"].tolist()
    colors  = colour_for(len(classes))
    axes[0].barh(classes, summary["Total objects"], color=colors)
    axes[0].set_xlabel("Total objects"); axes[0].set_title("Total objects per class")
    axes[0].invert_yaxis()
    for s in ["top","right"]: axes[0].spines[s].set_visible(False)
    axes[1].barh(classes, summary["Images w/ class"], color=colors)
    axes[1].set_xlabel("Number of images"); axes[1].set_title("Images containing each class")
    axes[1].invert_yaxis()
    for s in ["top","right"]: axes[1].spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, out / "01_class_counts.png", ds_name, ds_path)


def plot_avg_objects(summary, out, ds_name, ds_path):
    classes = summary["Class"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Average object count per image  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.bar(classes, summary["Avg objects / image"], color=colour_for(len(classes)))
    ax.set_ylabel("Avg objects / image\n(images without class excluded)")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "02_avg_objects_per_image.png", ds_name, ds_path)


def plot_avg_area(summary, out, ds_name, ds_path):
    classes = summary["Class"].tolist()
    colors  = colour_for(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Average area coverage per class  —  {ds_name}", fontsize=12, fontweight="bold")
    axes[0].bar(classes, summary["Avg area / image (px²)"], color=colors)
    axes[0].set_ylabel("px²"); axes[0].set_title("Avg total area per image (px²)")
    axes[0].set_xticklabels(classes, rotation=30, ha="right")
    for s in ["top","right"]: axes[0].spines[s].set_visible(False)
    axes[1].bar(classes, summary["Avg area % / image"], color=colors)
    axes[1].set_ylabel("% of image area"); axes[1].set_title("Avg % image area covered (per class)")
    axes[1].set_xticklabels(classes, rotation=30, ha="right")
    for s in ["top","right"]: axes[1].spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "03_avg_area.png", ds_name, ds_path)


def plot_exclusive_images(summary, out, ds_name, ds_path):
    classes = summary["Class"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Images containing ONLY one class  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.bar(classes, summary["Images w/ ONLY this class"], color=colour_for(len(classes)))
    ax.set_ylabel("Image count")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "04_exclusive_images.png", ds_name, ds_path)


def plot_cooccurrence(cooc, out, ds_name, ds_path):
    fig, ax = plt.subplots(figsize=(max(7, len(cooc)*0.9), max(6, len(cooc)*0.8)))
    fig.suptitle(f"Class co-occurrence matrix  —  {ds_name}", fontsize=12, fontweight="bold")
    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "# images"}, square=True)
    ax.set_title("# images containing both classes simultaneously", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "05_cooccurrence_matrix.png", ds_name, ds_path)


def plot_split_distribution(image_df, out, ds_name, ds_path):
    split_counts = image_df.groupby("split").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(f"Images per dataset split  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.bar(split_counts.index, split_counts.values, color=colour_for(len(split_counts)))
    for i, (_, v) in enumerate(split_counts.items()):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of images")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "06_split_distribution.png", ds_name, ds_path)


def plot_objects_per_image_hist(flat, out, ds_name, ds_path):
    obj_counts = flat.groupby("image").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"Distribution of total objects per image  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.hist(obj_counts, bins=range(1, obj_counts.max()+2),
            color=PALETTE[0], edgecolor="white", align="left")
    ax.set_xlabel("Number of objects in image"); ax.set_ylabel("Number of images")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "07_objects_per_image_distribution.png", ds_name, ds_path)


def plot_area_boxplot(flat, out, ds_name, ds_path):
    classes = sorted(flat["category"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.2), 5))
    fig.suptitle(f"Individual object area distribution per class  —  {ds_name}", fontsize=12, fontweight="bold")
    data = [flat[flat["category"] == c]["obj_area"].values for c in classes]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colour_for(len(classes))):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(classes)+1))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Object area (px²)")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "08_area_boxplot.png", ds_name, ds_path)


def plot_class_mix(image_df, out, ds_name, ds_path):
    df2 = image_df.copy()
    df2["n_classes"] = df2["categories"].apply(len)
    counts = df2["n_classes"].value_counts().sort_index()
    labels = [f"{k} class{'es' if k>1 else ''}" for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"Images by number of distinct classes present  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.bar(labels, counts.values, color=PALETTE[:len(labels)])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom")
    ax.set_ylabel("Number of images")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "09_classes_per_image.png", ds_name, ds_path)


def plot_stacked_split_class(flat, out, ds_name, ds_path):
    pivot = flat.groupby(["split","category"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.T.plot(kind="bar", stacked=False, ax=ax, color=PALETTE[:len(pivot)])
    fig.suptitle(f"Object count per class by split  —  {ds_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Number of objects")
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.legend(title="Split")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "10_objects_by_split_and_class.png", ds_name, ds_path)


def plot_object_area_violin(flat, out, ds_name, ds_path):
    df = flat.copy()
    df["log_area"] = np.log1p(df["obj_area"])
    classes = sorted(df["category"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.2), 5))
    fig.suptitle(f"Object area distribution per class (log scale)  —  {ds_name}", fontsize=12, fontweight="bold")
    parts = ax.violinplot(
        [df[df["category"]==c]["log_area"].values for c in classes],
        positions=range(len(classes)), showmedians=True
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("log(1 + area)  [px²]")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_fig(fig, out / "11_area_violin.png", ds_name, ds_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Text outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_class_image_lists(lists, out, ds_name, ds_path):
    lines = [
        f"Dataset   : {ds_name}",
        f"Full path : {ds_path}",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Content   : Per-class image lists",
        "          : 'All'  = images containing ≥1 object of that class",
        "          : 'Only' = images whose every object belongs to that class",
        "=" * 70,
    ]
    for cls, data in lists.items():
        lines += [
            f"\n{'='*70}",
            f"CLASS: {cls}",
            f"{'='*70}",
            f"\n-- All images containing '{cls}'  ({len(data['all'])}) --",
        ]
        lines += [f"  {img}" for img in data["all"]]
        lines.append(f"\n-- Images containing ONLY '{cls}'  ({len(data['only'])}) --")
        lines += ([f"  {img}" for img in data["only"]] if data["only"] else ["  (none)"])
    with open(out / "class_image_lists.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dataset-level analysis")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out",  default="./dataset_report_")
    parser.add_argument("--name", default=None,
                        help="Dataset name to use in outputs (overrides auto-detection)")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_name, ds_path = get_dataset_info(root)
    if args.name:      ds_name = args.name
    print(f"Dataset : {ds_name}")
    print(f"Path    : {ds_path}")

    print("\nLoading JSON annotations …")
    records  = load_jsons(root)
    print(f"  → {len(records)} images found\n")

    flat     = build_flat_df(records)
    image_df = build_image_df(records)

    # 1. Class summary
    print("Building class summary …")
    summary = class_summary_table(flat, image_df)
    write_csv_with_header(
        summary, out / "class_summary.csv", ds_name, ds_path,
        "Per-class statistics: object counts, image counts, area coverage, exclusivity"
    )
    print(summary.to_string(index=False))

    # 2. Co-occurrence
    print("\nBuilding co-occurrence matrix …")
    cooc = cooccurrence_matrix(image_df)
    write_csv_with_header(
        cooc.reset_index().rename(columns={"index": "Class\\Class"}),
        out / "cooccurrence_matrix.csv", ds_name, ds_path,
        "Co-occurrence: number of images containing both row-class AND column-class"
    )
    print(cooc.to_string())

    # 3. Image lists
    print("\nGenerating per-class image lists …")
    lists = class_image_lists(flat, image_df)
    save_class_image_lists(lists, out, ds_name, ds_path)

    # 4. Stats JSON
    n_images    = len(image_df)
    n_labeled   = int((image_df["n_objects"] > 0).sum())
    obj_per_img = flat.groupby("image").size()
    stats = {
        "dataset_name":           ds_name,
        "dataset_path":           ds_path,
        "generated":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images":           n_images,
        "total_objects":          len(flat),
        "images_with_objects":    n_labeled,
        "images_without_objects": n_images - n_labeled,
        "pct_images_labeled":     round(100 * n_labeled / n_images, 1) if n_images else 0,
        "avg_objects_per_image":  round(obj_per_img.mean(), 2),
        "median_objects_per_img": round(obj_per_img.median(), 2),
        "max_objects_in_image":   int(obj_per_img.max()),
        "total_classes":          int(flat["category"].nunique()),
        "classes":                sorted(flat["category"].unique().tolist()),
    }
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nDataset overview:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 5. Plots
    print("\nGenerating plots …")
    plot_class_counts(summary, out, ds_name, ds_path)
    plot_avg_objects(summary, out, ds_name, ds_path)
    plot_avg_area(summary, out, ds_name, ds_path)
    plot_exclusive_images(summary, out, ds_name, ds_path)
    plot_cooccurrence(cooc, out, ds_name, ds_path)
    plot_split_distribution(image_df, out, ds_name, ds_path)
    plot_objects_per_image_hist(flat, out, ds_name, ds_path)
    plot_area_boxplot(flat, out, ds_name, ds_path)
    plot_class_mix(image_df, out, ds_name, ds_path)
    plot_stacked_split_class(flat, out, ds_name, ds_path)
    plot_object_area_violin(flat, out, ds_name, ds_path)

    print(f"\n✓ All outputs saved to: {out.resolve()}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
