# Journal-2 — CNN Ensemble for Concrete Bridge Damage Detection

A PyTorch-based repository for training and evaluating CNN ensemble methods
for multi-class concrete damage classification. Designed as the first stage
of a two-stage bridge inspection pipeline: this repository classifies images
and flags damaged ones; a downstream segmentation repository then generates
pixel-level damage masks.

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
│   ├── train_config.yaml       ← Hyperparameters, paths, augmentation,
│   │                              class balancing settings
│   ├── ensemble_config.yaml    ← Which models and fusion methods to run
│   └── deploy_config.yaml      ← Deployment inference settings
│
├── src/
│   ├── models/
│   │   ├── base_model.py       ← Shared training loop, checkpoint logic,
│   │   │                          validation and test evaluation
│   │   ├── vgg16.py            ← VGG16 architecture loader
│   │   ├── resnet50.py         ← ResNet50 architecture loader
│   │   └── alexnet.py          ← AlexNet architecture loader
│   │
│   ├── ensemble/
│   │   ├── hard_voting.py      ← Majority vote over model predictions
│   │   ├── soft_voting.py      ← Weighted/unweighted probability averaging
│   │   ├── bayesian_fusion.py  ← Sequential Bayesian updating
│   │   ├── sugeno_fuzzy.py     ← Sugeno fuzzy integral fusion
│   │   └── mlp_meta_learner.py ← Trained MLP stacking meta-learner
│   │
│   ├── data/
│   │   ├── dataloader.py       ← Dataset class, DataLoader factory,
│   │   │                          class distribution reporting,
│   │   │                          WeightedRandomSampler for class balancing
│   │   └── augmentations.py    ← Training and evaluation transforms
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← Accuracy, Precision, Recall, F1, F2,
│   │   │                          Specificity (overall + per class)
│   │   ├── confusion_matrix.py ← Confusion matrix plotting and saving,
│   │   │                          multi-method grid comparison plot
│   │   └── plots.py            ← Training curves, ensemble comparison
│   │                              bar charts, per-class F1 comparison
│   │
│   └── utils/
│       ├── io_utils.py         ← File I/O, config loading, timestamped
│       │                          run directory management, CSV/JSON helpers
│       └── logger.py           ← Logging setup (console + run.log file)
│
├── scripts/
│   ├── train.py                ← Train a single CNN model
│   ├── evaluate.py             ← Run ensemble methods and compare results
│   ├── deploy.py               ← Inference on new unlabelled images
│   └── experiments/
│       └── bayesian_permutations.py  ← Compare all model orderings for
│                                        Sequential Bayesian fusion
│
├── saved_models/               ← Trained weights (.pth files) — not in Git
│   ├── vgg16/
│   │   ├── best.pth            ← Best validation accuracy checkpoint
│   │   └── last.pth            ← Most recent epoch checkpoint
│   ├── resnet50/
│   └── alexnet/
│
├── results/                    ← Generated outputs — not in Git
│   ├── training/               ← Per-model timestamped run folders
│   │   └── vgg16/
│   │       └── 2026-03-19_14-32/
│   │           ├── predictions/   ← val + test CSVs (used by ensemble)
│   │           ├── metrics/       ← test_metrics.json
│   │           ├── curves/        ← loss and accuracy PNGs
│   │           ├── confusion_matrices/
│   │           └── run.log
│   ├── ensemble/               ← Ensemble comparison outputs
│   │   └── comparison/
│   │       └── 2026-03-19_15-00/
│   │           ├── metrics/       ← per-method JSONs + summary CSV
│   │           ├── confusion_matrices/
│   │           ├── plots/
│   │           └── run.log
│   └── experiments/
│       └── bayesian_permutations/
│
├── run_pipeline.py             ← Run training + ensemble in one command
├── pipeline.yaml               ← Two-stage inspection pipeline config
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
Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the
correct command for your system. For CUDA 12.4:
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
Open `configs/train_config.yaml` and update the `paths` section:
```yaml
paths:
  train:       "C:/your/path/to/data/train"
  val:         "C:/your/path/to/data/val"
  test:        "C:/your/path/to/data/test"
  saved_models: "saved_models"
  results:     "results/training"
```

---

## Usage

### Option A — Run the full pipeline (recommended)
Trains all models sequentially then runs ensemble evaluation in one command.
Edit the configuration block at the top of `run_pipeline.py`, then:
```bash
python run_pipeline0.py
```

To run only training or only the ensemble step:
```bash
# Edit MODE = "train_only" or MODE = "ensemble_only" in run_pipeline0.py
python run_pipeline0.py
```

To skip a specific model, set `"enabled": False` in the `MODELS` list
inside `run_pipeline.py`.

### Option B — Run scripts individually

**Train one model:**
```bash
python scripts/train.py --model vgg16    --config configs/train_config.yaml
python scripts/train.py --model resnet50 --config configs/train_config.yaml
python scripts/train.py --model alexnet  --config configs/train_config.yaml
```
Add `--show_plots` to display training curves interactively.

**Run ensemble evaluation** (requires all models to have been trained):
```bash
python scripts/ensemble_eval.py --config configs/ensemble_config.yaml

# Include MLP meta-learner (trained on validation predictions):
python scripts/ensemble_eval.py --config configs/ensemble_config.yaml --train_mlp
```

**Deploy on new images:**
```bash
python scripts/deploy.py --input path/to/images \
                         --config configs/deploy_config.yaml
```

**Bayesian ordering experiment** (compare all 6 model orderings):
```bash
python scripts/experiments/bayesian_permutations.py
```

---

## Training outputs

Each training run saves to a timestamped folder:
```
results/training/vgg16/2026-03-19_14-32/
    predictions/
        predictions.csv        ← best validation epoch (used by ensemble)
        test_predictions.csv   ← test set predictions
    metrics/
        test_metrics.json      ← accuracy, F1, F2, per-class metrics
    curves/
        vgg16_loss_curve.png
        vgg16_accuracy_curve.png
    confusion_matrices/
        confusion_matrix.png
    run.log
```

The class distribution of your dataset is printed automatically at the
start of every training run, along with an imbalance warning if needed.

---

## Class balancing

If your dataset has significantly more images in some classes than others,
enable weighted sampling in `configs/train_config.yaml`:
```yaml
balancing:
  balanced_sampling: true
```

This oversamples underrepresented classes during training without creating
new images. It can be combined with data augmentation for best results.
Evaluation sets are never rebalanced — metrics always reflect the real
class distribution.

---

## Ensemble methods

| Method | Description |
|--------|-------------|
| Hard Voting | Each model votes for its top class; majority wins |
| Soft Voting | Weighted/unweighted average of probability distributions |
| Sequential Bayesian | Each model sequentially updates the posterior of the previous one |
| Sugeno Fuzzy Integral | Fuzzy measure-based aggregation capturing model interactions |
| MLP Meta-Learner | Small neural network trained on stacked model outputs (stacking) |

The `bayesian_permutations.py` experiment evaluates all 6 possible model
orderings for Sequential Bayesian fusion and compares them against the
voting methods to identify the optimal ordering.

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
