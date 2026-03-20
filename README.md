# Journal-2 — CNN Ensemble for Concrete Bridge Damage Detection

A PyTorch-based repository for training and evaluating CNN ensemble methods for multi-class concrete damage classification. Designed as the first stage of a two-stage bridge inspection pipeline: this repository classifies images and flags damaged ones; a downstream segmentation repository then generates pixel-level masks.

---

## Damage Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `crack` | Surface or structural cracks |
| 1 | `efflorescence` | White salt deposits on concrete surface |
| 2 | `spalling` | Concrete surface detachment |
| 3 | `undamaged` | No visible damage |

---

## Repository Structure

```
Journal-2/
│
├── configs/
│   ├── train_config.yaml       ← Hyperparameters, paths, augmentation settings
│   ├── ensemble_config.yaml    ← Which models and fusion methods to run
│   └── deploy_config.yaml      ← Deployment inference settings
│
├── src/
│   ├── models/
│   │   ├── base_model.py       ← Shared training loop, checkpoint logic
│   │   ├── vgg16.py            ← VGG16 architecture loader
│   │   ├── resnet50.py         ← ResNet50 architecture loader
│   │   └── alexnet.py          ← AlexNet architecture loader
│   │
│   ├── ensemble/
│   │   ├── hard_voting.py      ← Majority vote over model predictions
│   │   ├── soft_voting.py      ← Average of model probability distributions
│   │   ├── bayesian_fusion.py  ← Sequential Bayesian updating
│   │   ├── sugeno_fuzzy.py     ← Sugeno fuzzy integral fusion
│   │   └── mlp_meta_learner.py ← Trained MLP stacking meta-learner
│   │
│   ├── data/
│   │   ├── dataloader.py       ← Dataset class and DataLoader factory
│   │   └── augmentations.py    ← Training and evaluation transforms
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← Accuracy, Precision, Recall, F1, F2, Specificity
│   │   ├── confusion_matrix.py ← Confusion matrix plotting and saving
│   │   └── plots.py            ← Training curves and comparison bar charts
│   │
│   └── utils/
│       ├── io_utils.py         ← File I/O, config loading, run directory management
│       └── logger.py           ← Logging setup (console + file)
│
├── scripts/
│   ├── train.py                ← Train a single CNN model
│   ├── evaluate.py             ← Run ensemble methods and compare results
│   └── deploy.py               ← Inference on new unlabelled images
│
├── saved_models/               ← Trained weights (.pth files) — not in Git
│   ├── vgg16/
│   ├── resnet50/
│   └── alexnet/
│
├── results/                    ← Generated outputs — not in Git
│   ├── training/               ← Per-model, per-run metrics, curves, matrices
│   └── ensemble/               ← Ensemble comparison outputs and predictions
│
├── pipeline.yaml               ← Top-level two-stage inspection pipeline config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/JuanAvendano/Journal-2.git
cd Journal-2
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA support
Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command. For CUDA 12.4:
```bash
pip install torch==2.10.1 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Prepare your dataset
Organise your images in the following folder structure:
```
data/
├── train/
│   ├── crack/
│   ├── efflorescence/
│   ├── spalling/
│   └── undamaged/
├── val/
│   └── (same subfolders)
└── test/
    └── (same subfolders)
```

### 5. Update paths in the config
Open `configs/train_config.yaml` and update the `paths` section to point to your dataset:
```yaml
paths:
  train: "C:/your/path/to/data/train"
  val:   "C:/your/path/to/data/val"
  test:  "C:/your/path/to/data/test"
```

---

## Usage

### Training a model
```bash
python scripts/train.py --model vgg16    --config configs/train_config.yaml
python scripts/train.py --model resnet50 --config configs/train_config.yaml
python scripts/train.py --model alexnet  --config configs/train_config.yaml
```

Add `--show_plots` to display training curves interactively.

Each run saves results to a timestamped folder:
```
results/training/vgg16/2026-03-19_14-32/
    predictions/predictions.csv      ← used by ensemble methods
    predictions/test_predictions.csv
    metrics/test_metrics.json
    curves/vgg16_loss_curve.png
    curves/vgg16_accuracy_curve.png
    confusion_matrices/confusion_matrix.png
    run.log
```

### Evaluating ensemble methods
```bash
python scripts/evaluate.py --config configs/ensemble_config.yaml

# Include the MLP meta-learner (trains it on validation predictions):
python scripts/evaluate.py --config configs/ensemble_config.yaml --train_mlp
```

### Deployment (inference on new images)
```bash
python scripts/deploy.py --input path/to/new/images --config configs/deploy_config.yaml
```

Output is a JSON file listing flagged images with their predicted damage class and confidence score. This JSON is consumed by the segmentation pipeline.

---

## Ensemble Methods

| Method | Description |
|--------|-------------|
| Hard Voting | Each model votes for its top class; majority wins |
| Soft Voting | Average of class probability distributions |
| Sequential Bayesian | Sequential Bayesian update: each model updates the posterior left by the previous one |
| Sugeno Fuzzy Integral | Fuzzy measure-based aggregation capturing model interactions |
| MLP Meta-Learner | Small neural network trained to combine model outputs (stacking) |

---

## Environment

Developed and tested with:
- Python 3.10
- PyTorch 2.10.1 + CUDA 12.4
- torchvision 0.25.0
- Windows 10
- NVIDIA GeForce GTX 1060 (6GB)

---

## Citation

*To be added upon publication.*
