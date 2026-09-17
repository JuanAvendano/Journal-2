# =============================================================
# Vendi Tools - Work From Saved File Paths
# =============================================================
#
# Use this when you already have a list of image file paths
# (e.g. a JSON saved by vendi_subset_pipeline.py, or your own
# list) and want to:
#   - extract features for just those paths
#   - compute the Vendi Score of that exact set
#   - optionally re-select a tighter subset from them
#   - copy them into a new folder
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
# SECTION 1 - Feature Extractor (ResNet50)  [same as pipeline]
# =============================================================
class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def extract(self, image_paths, batch_size=32, verbose=True):
        features, total = [], len(image_paths)
        for i in range(0, total, batch_size):
            batch = [self.preprocess(self._load_image(p))
                     for p in image_paths[i:i + batch_size]]
            batch = torch.stack(batch).to(self.device)
            out = self.model(batch).squeeze(-1).squeeze(-1)
            features.append(out.cpu().numpy())
            if verbose:
                print(f"      features: {min(i+batch_size, total)}/{total}",
                      end="\r")
        if verbose:
            print()
        return np.vstack(features)


# =============================================================
# SECTION 2 - Vendi Score
# =============================================================
def vendi_score(K):
    n = K.shape[0]
    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = eigvals[eigvals > 1e-12]
    return float(np.exp(-np.sum(eigvals * np.log(eigvals))))


def vendi_from_embeddings(X):
    return vendi_score(cosine_similarity(X))


# =============================================================
# SECTION 3 - Path I/O
# =============================================================
def load_paths(source):
    """
    Load file paths from:
      - a .json file (list of paths)
      - a .txt file (one path per line)
      - a Python list (returned as-is)
    """
    if isinstance(source, list):
        paths = source
    elif source.lower().endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            paths = json.load(f)
    elif source.lower().endswith(".txt"):
        with open(source, "r", encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    else:
        raise ValueError("Provide a .json, .txt, or a Python list.")

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"      WARNING: {len(missing)} paths do not exist on disk.")
    return paths


def save_paths(paths, json_path, copy_to=None):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)
    print(f"      saved {len(paths)} paths -> {json_path}")
    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for p in paths:
            shutil.copy2(p, os.path.join(copy_to, os.path.basename(p)))
        print(f"      copied {len(paths)} images -> {copy_to}")


# =============================================================
# SECTION 4 - Subset selection (same methods as pipeline)
# =============================================================
def select_densest_cluster(X, n_select, n_clusters=10, random_state=42):
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
        raise ValueError(f"No cluster has >= {n_select} images. Lower n_clusters.")
    best = int(np.argmin(densities))
    member_idx = np.where(labels == best)[0]
    dists = np.linalg.norm(X[member_idx] - centroids[best], axis=1)
    return member_idx[np.argsort(dists)[:n_select]]


def select_target_vendi(X, n_select, target=25.0, tol=1.5,
                        max_iter=12, random_state=42):
    n = X.shape[0]
    lo, hi = 1, max(2, n // n_select)
    best = None
    for it in range(max_iter):
        mid = max(1, (lo + hi) // 2)
        if mid == 1:
            km = KMeans(n_clusters=1, random_state=random_state, n_init=10)
            km.fit(X)
            d = np.linalg.norm(X - km.cluster_centers_[0], axis=1)
            idx = np.argsort(d)[:n_select]
        else:
            try:
                idx = select_densest_cluster(X, n_select, mid, random_state)
            except ValueError:
                hi = mid - 1
                continue
        v = vendi_from_embeddings(X[idx])
        print(f"      [iter {it+1}] n_clusters={mid:>3}  Vendi={v:7.3f}  target={target}")
        if best is None or abs(v - target) < abs(best[1] - target):
            best = (idx, v, mid)
        if abs(v - target) <= tol:
            break
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
# SECTION 5 - Top-level helpers
# =============================================================
def score_paths(source, batch_size=32):
    """Compute the Vendi Score of an existing list/file of paths."""
    print("[1/2] Loading paths...")
    paths = load_paths(source)
    print(f"      {len(paths)} paths.")
    print("[2/2] Extracting features + scoring...")
    X = FeatureExtractor().extract(paths, batch_size=batch_size)
    v = vendi_from_embeddings(X)
    print(f"      Vendi Score: {v:.4f}")
    return v, paths, X


def reselect_from_paths(source, n_select=2000, method="target_vendi",
                        target=25.0, batch_size=32,
                        save_json=None, copy_to=None, **kwargs):
    """
    Take an existing set of paths, extract features once, and pick a
    tighter subset from them (without re-touching the full dataset).
    """
    v, paths, X = score_paths(source, batch_size=batch_size)

    print(f"      Selecting via '{method}'...")
    if method == "target_vendi":
        idx, _, _ = select_target_vendi(X, n_select, target, **kwargs)
    elif method == "densest_cluster":
        idx = select_densest_cluster(X, n_select, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    selected = [paths[i] for i in idx]
    print(f"      Subset size: {len(selected)}   "
          f"Subset Vendi: {vendi_from_embeddings(X[idx]):.4f}")

    if save_json:
        save_paths(selected, save_json, copy_to=copy_to)
    return selected


# =============================================================
# Example usage
# =============================================================
if __name__ == "__main__":
    SELECTION = r"C:\path\to\subset_target25.json"

    # ---- A) Just score an existing list of paths ----
    score_paths(SELECTION, batch_size=32)

    # ---- B) Re-select a tighter subset from existing paths ----
    # reselect_from_paths(
    #     SELECTION,
    #     n_select=1000,
    #     method="target_vendi",
    #     target=18.0,
    #     save_json=r"C:\path\to\subset_target18.json",
    #     copy_to=r"C:\path\to\efflorescence_subset_18",
    # )

    # ---- C) Score your own custom list ----
    # my_paths = [r"C:\imgs\a.png", r"C:\imgs\b.png", r"C:\imgs\c.png"]
    # score_paths(my_paths)
