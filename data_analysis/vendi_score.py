# =============================================================
# Vendi Score with CNN Feature Extraction
# =============================================================

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================
# SECTION 1 — Feature Extractor (ResNet50 penultimate layer)
# =============================================================
class FeatureExtractor:
    """
    Extracts deep feature embeddings from images using a
    pretrained CNN (ResNet50), removing the classification head.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # remove final FC layer -> output is 2048-dim pooled feature
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval().to(self.device)

        # ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_image(self, path):
        """Load an image with cv2 and convert BGR -> RGB."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @torch.no_grad()
    def extract(self, image_paths, batch_size=32):
        """
        Extract embeddings for a list of image paths.

        Returns:
            np.ndarray: (n_samples, 2048) feature matrix.
        """
        features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = []

            for p in batch_paths:
                img = self._load_image(p)
                tensor = self.preprocess(img)
                batch.append(tensor)

            batch = torch.stack(batch).to(self.device)
            out = self.model(batch)               # (B, 2048, 1, 1)
            out = out.squeeze(-1).squeeze(-1)      # (B, 2048)
            features.append(out.cpu().numpy())

        return np.vstack(features)


# =============================================================
# SECTION 2 — Image Folder Loader
# =============================================================
def gather_image_paths(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif")):
    """
    Recursively collect all image file paths from a folder.
    """
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return sorted(paths)


# =============================================================
# SECTION 3 — Vendi Score
# =============================================================
def vendi_score(K):
    """
    Compute the Vendi Score from a similarity matrix K.
    """
    n = K.shape[0]
    K_norm = K / n
    eigvals = np.linalg.eigvalsh(K_norm)
    eigvals = eigvals[eigvals > 1e-12]
    entropy = -np.sum(eigvals * np.log(eigvals))
    return np.exp(entropy)


def vendi_from_embeddings(X, similarity="cosine"):
    """
    Compute the Vendi Score directly from feature vectors.
    """
    if similarity == "cosine":
        K = cosine_similarity(X)
    else:
        raise ValueError("Unsupported similarity type.")
    return vendi_score(K)


# =============================================================
# SECTION 4 — End-to-End Pipeline
# =============================================================
def compute_dataset_vendi(folder, batch_size=32):
    """
    Full pipeline: image folder -> embeddings -> Vendi Score.
    """
    print(f"[1/3] Gathering images from: {folder}")
    paths = gather_image_paths(folder)
    print(f"      Found {len(paths)} images.")

    print("[2/3] Extracting CNN features...")
    extractor = FeatureExtractor()
    X = extractor.extract(paths, batch_size=batch_size)
    print(f"      Feature matrix shape: {X.shape}")

    print("[3/3] Computing Vendi Score...")
    score = vendi_from_embeddings(X)
    print(f"      Vendi Score: {score:.4f}")

    return score


# =============================================================
# Example usage
# =============================================================
if __name__ == "__main__":
    dataset_folder = r"D:\JCA\07-Data\01_Concrete\03-experiment_datasets\A\A6\02-validation\crack"
    compute_dataset_vendi(dataset_folder, batch_size=32)