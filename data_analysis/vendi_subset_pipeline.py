# =============================================================
# Vendi Score Pipeline
# Feature Extraction  ->  Vendi Scoring  ->  Subset Selection
# =============================================================
#
# Selection modes:
#   1. densest_cluster      - tightest k-means cluster (low Vendi)
#   2. highest_similarity   - most 'central' images   (low Vendi)
#   3. greedy_minimize      - greedily minimize Vendi  (slow, precise)
#   4. target_vendi         - binary-search a subset to hit a target
#
# =============================================================

import os
import json
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================
# SECTION 1 - Feature Extractor (ResNet50 penultimate layer)
# =============================================================
class FeatureExtractor:
    """
    Extracts 2048-dim deep features from images using a
    pretrained ResNet50 with the classification head removed.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval().to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_image(self, path):
        """Load image with cv2, BGR -> RGB."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def extract(self, image_paths, batch_size=32, verbose=True):
        """
        Extract embeddings for a list of image paths.
        Returns (n_samples, 2048) feature matrix.
        """
        features = []
        total = len(image_paths)

        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = [self.preprocess(self._load_image(p)) for p in batch_paths]
            batch = torch.stack(batch).to(self.device)

            out = self.model(batch).squeeze(-1).squeeze(-1)   # (B, 2048)
            features.append(out.cpu().numpy())

            if verbose:
                done = min(i + batch_size, total)
                print(f"      features: {done}/{total}", end="\r")

        if verbose:
            print()
        return np.vstack(features)


# =============================================================
# SECTION 2 - Image Folder Loader
# =============================================================
def gather_image_paths(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif")):
    """Recursively collect all image file paths from a folder."""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return sorted(paths)


# =============================================================
# SECTION 3 - Vendi Score
# =============================================================
def vendi_score(K):
    """Vendi Score from an (n, n) cosine-similarity matrix."""
    n = K.shape[0]
    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = eigvals[eigvals > 1e-12]
    return float(np.exp(-np.sum(eigvals * np.log(eigvals))))


def vendi_from_embeddings(X):
    """Vendi Score directly from a feature matrix."""
    return vendi_score(cosine_similarity(X))


# =============================================================
# SECTION 4 - Subset Selection Methods
# =============================================================

# -------------------------------------------------------------
# 4.1  Densest cluster around a centroid  (low Vendi)
# -------------------------------------------------------------
def select_densest_cluster(X, n_select=2000, n_clusters=10, random_state=42):
    """
    Cluster features, find the tightest cluster big enough to fill
    n_select, return the n_select images closest to its centroid.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    densities = []
    for c in range(n_clusters):
        members = X[labels == c]
        if len(members) < n_select:
            densities.append(np.inf)
            continue
        densities.append(np.linalg.norm(members - centroids[c], axis=1).mean())

    if np.all(np.isinf(densities)):
        raise ValueError(
            f"No cluster has >= {n_select} images. Lower n_clusters."
        )

    best = int(np.argmin(densities))
    member_idx = np.where(labels == best)[0]
    dists = np.linalg.norm(X[member_idx] - centroids[best], axis=1)
    return member_idx[np.argsort(dists)[:n_select]]


# -------------------------------------------------------------
# 4.2  Highest mean similarity  (crowded center, low Vendi)
# -------------------------------------------------------------
def select_highest_similarity(X, n_select=2000):
    """Keep the n_select images with highest mean similarity to the rest."""
    K = cosine_similarity(X)
    np.fill_diagonal(K, 0.0)
    mean_sim = K.mean(axis=1)
    return np.argsort(mean_sim)[::-1][:n_select]


# -------------------------------------------------------------
# 4.3  Greedy Vendi minimization  (precise, slow)
# -------------------------------------------------------------
def select_greedy_minimize(X, n_select=2000, seed_strategy="central",
                           random_state=42):
    """
    Greedily grow a subset, each step adding the image that keeps the
    running Vendi Score lowest. O(n_select * n) similarity work.
    Practical for a few thousand selections; not for huge n_select.
    """
    n = X.shape[0]
    K = cosine_similarity(X)

    if seed_strategy == "central":
        np.fill_diagonal(K, 0.0)
        start = int(np.argmax(K.mean(axis=1)))
        np.fill_diagonal(K, 1.0)
    else:
        rng = np.random.default_rng(random_state)
        start = int(rng.integers(n))

    selected = [start]
    remaining = set(range(n)) - {start}

    while len(selected) < n_select:
        best_idx, best_v = None, np.inf
        # evaluate a candidate by the Vendi of selected + candidate
        for cand in remaining:
            idx = selected + [cand]
            sub = K[np.ix_(idx, idx)]
            v = vendi_score(sub)
            if v < best_v:
                best_v, best_idx = v, cand
        selected.append(best_idx)
        remaining.discard(best_idx)
        print(f"      greedy: {len(selected)}/{n_select}  Vendi={best_v:.3f}",
              end="\r")
    print()
    return np.array(selected)


# -------------------------------------------------------------
# 4.4  Target Vendi Score  (binary search on cluster tightness)
# -------------------------------------------------------------
def select_target_vendi(X, n_select=2000, target=25.0, tol=1.5,
                        max_iter=12, random_state=42):
    """
    Find a subset of size n_select whose Vendi Score is close to `target`.

    Strategy: sweep the number of k-means clusters. More clusters -> each
    cluster is tighter -> lower subset Vendi. We binary-search n_clusters
    to land within `tol` of the target.

    Args:
        target (float): desired Vendi Score (e.g. 25).
        tol (float): acceptable absolute deviation from target.
        max_iter (int): max search iterations.
    Returns:
        (indices, achieved_vendi, n_clusters_used)
    """
    n = X.shape[0]

    # n_clusters must allow at least one cluster to hold n_select images.
    # More clusters => smaller/tighter clusters => lower Vendi.
    lo, hi = 1, max(2, n // n_select)   # upper bound keeps clusters fillable
    best = None

    for it in range(max_iter):
        mid = (lo + hi) // 2
        mid = max(1, mid)

        if mid == 1:
            # single cluster = densest overall blob of n_select
            km = KMeans(n_clusters=1, random_state=random_state, n_init=10)
            km.fit(X)
            d = np.linalg.norm(X - km.cluster_centers_[0], axis=1)
            idx = np.argsort(d)[:n_select]
        else:
            try:
                idx = select_densest_cluster(X, n_select, mid, random_state)
            except ValueError:
                # too many clusters to fill n_select -> back off
                hi = mid - 1
                continue

        v = vendi_from_embeddings(X[idx])
        print(f"      [iter {it+1}] n_clusters={mid:>3}  Vendi={v:7.3f}  "
              f"target={target}")

        if best is None or abs(v - target) < abs(best[1] - target):
            best = (idx, v, mid)

        if abs(v - target) <= tol:
            break

        # higher Vendi than target -> need tighter clusters -> more clusters
        if v > target:
            lo = mid + 1
        else:
            hi = mid - 1

        if lo > hi:
            break

    idx, v, mid = best
    print(f"      best: Vendi={v:.3f} (target {target}) with n_clusters={mid}")
    return idx, v, mid


# =============================================================
# SECTION 5 - Save / Load selected paths
# =============================================================
def save_selection(selected_paths, json_path, copy_to=None):
    """
    Save selected file paths to JSON. Optionally copy the images
    into a new folder (preserving filenames).
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected_paths, f, indent=2, ensure_ascii=False)
    print(f"      saved {len(selected_paths)} paths -> {json_path}")

    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for p in selected_paths:
            shutil.copy2(p, os.path.join(copy_to, os.path.basename(p)))
        print(f"      copied {len(selected_paths)} images -> {copy_to}")


def load_selection(json_path):
    """Load a previously saved list of file paths."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================
# SECTION 6 - End-to-End Pipeline
# =============================================================
def build_subset(folder, n_select=2000, method="target_vendi",
                 target=25.0, batch_size=32, save_json=None,
                 copy_to=None, **kwargs):
    """
    Full pipeline: image folder -> features -> subset -> save.

    method: 'densest_cluster' | 'highest_similarity' |
            'greedy_minimize' | 'target_vendi'
    """
    print(f"[1/4] Gathering images from: {folder}")
    paths = gather_image_paths(folder)
    print(f"      Found {len(paths)} images.")

    print("[2/4] Extracting CNN features...")
    X = FeatureExtractor().extract(paths, batch_size=batch_size)
    print(f"      Feature matrix: {X.shape}")
    print(f"      Full-set Vendi: {vendi_from_embeddings(X):.4f}")

    print(f"[3/4] Selecting subset via '{method}'...")
    if method == "densest_cluster":
        idx = select_densest_cluster(X, n_select, **kwargs)
    elif method == "highest_similarity":
        idx = select_highest_similarity(X, n_select)
    elif method == "greedy_minimize":
        idx = select_greedy_minimize(X, n_select, **kwargs)
    elif method == "target_vendi":
        idx, achieved, _ = select_target_vendi(X, n_select, target, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    subset_vendi = vendi_from_embeddings(X[idx])
    selected_paths = [paths[i] for i in idx]
    print(f"      Subset size: {len(idx)}   Subset Vendi: {subset_vendi:.4f}")

    print("[4/4] Saving selection...")
    if save_json:
        save_selection(selected_paths, save_json, copy_to=copy_to)

    return selected_paths, subset_vendi


# =============================================================
# Example usage
# =============================================================
if __name__ == "__main__":
    DATASET = r"D:\JCA\07-Data\01_Concrete\02-total_compilation\Spalling\images"

    # ---- Target-Vendi subset for training (Vendi ~ 25) ----
    paths, v = build_subset(
        folder=DATASET,
        n_select=2000,
        method="target_vendi",
        target=25.0,
        batch_size=32,
        save_json=r"D:\JCA\07-Data\01_Concrete\03-experiment_datasets\A\A6\spalling_subset_target25.json",
        copy_to=r"D:\JCA\07-Data\01_Concrete\03-experiment_datasets\A\A6\01-train\spalling",   # set None to skip copy
    )
    print(f"\nDone. Final subset Vendi = {v:.4f}")

    # ---- Other methods (uncomment to use) ----
    # build_subset(DATASET, 2000, method="densest_cluster", n_clusters=8,
    #              save_json=r"C:\path\to\subset_dense.json")
    # build_subset(DATASET, 2000, method="highest_similarity",
    #              save_json=r"C:\path\to\subset_central.json")
    # build_subset(DATASET, 2000, method="greedy_minimize",
    #              save_json=r"C:\path\to\subset_greedy.json")