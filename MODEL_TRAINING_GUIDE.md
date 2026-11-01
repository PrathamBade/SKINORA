# SKINORA — ResNet50 Model Training Guide

This guide provides instructions for training, fine-tuning, and evaluating the ResNet50 acne-severity classification model on the **ACNE04** dataset.

---

## 1. Environment & Hardware Prerequisites

- **Python Version**: Python 3.10.x recommended
- **Operating System**: Windows 10/11, Linux, or macOS
- **Hardware Acceleration**:
  - NVIDIA GPU with CUDA support (Recommended for faster training, ~2-4 minutes)
  - CPU-only execution is fully supported (~12-15 minutes for 8 epochs)
- **Virtual Environment**:
  ```powershell
  cd Skinora
  # Create venv if not already present
  python -m venv venv
  .\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
  ```

---

## 2. Dataset Setup & Integrity

### Primary Dataset: ACNE04
- **Source**: [`Layered-Labs/ACNE04`](https://huggingface.co/datasets/Layered-Labs/ACNE04)
- **Size**: 1,457 facial photographs
- **Target Classes**:
  1. `0` — `mild`
  2. `1` — `moderate`
  3. `2` — `severe`
  4. `3` — `very_severe`
- **Class Distribution**:
  - `mild`: 513 total (35.2%)
  - `moderate`: 633 total (43.4%)
  - `severe`: 182 total (12.5%)
  - `very_severe`: 129 total (8.9%)
- **Data Splits**:
  - **Train partition**: 990 images (85% of train split)
  - **Validation partition**: 175 images (15% of train split, stratified)
  - **Test partition**: 292 held-out images (designated for final evaluation only)

### Optional Compatible Datasets
The training script supports optional compatible datasets via `--extra-data-dir`. The folder must follow this exact four-class structure:
```text
extra-data/
  mild/
  moderate/
  severe/
  very_severe/
```
> [!IMPORTANT]
> Do not merge dermoscopic datasets (e.g. HAM10000, ISIC) into acne severity training. They represent different visual modalities and disease labels.

---

## 3. Training Commands & Options

### Standard Training (Recommended)
Runs transfer learning with ImageNet-1K pretrained weights, layer-4 fine-tuning, balanced class-weighting, and Plateau LR scheduler:
```powershell
cd Skinora\backend
..\venv\Scripts\python.exe -m Skinora.backend.training.train_acne_model --experiment-id resnet50_optimized --epochs 8 --batch-size 16 --finetune-mode partial_layer4 --set-as-production
```

### Controlled Experiment Modes

| Experiment Flag | Fine-Tune Mode | Description |
|---|---|---|
| `--finetune-mode frozen` | Head Only | Freezes all ResNet50 backbone layers; trains only the final classification linear layer. |
| `--finetune-mode partial_layer4` | Layer 4 + Head | Default transfer learning. Fine-tunes high-level visual features while freezing generic lower filters. |
| `--finetune-mode deep_layer3_4` | Layer 3 + Layer 4 + Head | Deeper adaptation with differential learning rates (`--backbone-lr 1e-5`, `--head-lr 2e-4`). |
| `--finetune-mode full` | Full Backbone | Unfreezes all layers with low learning rate. Recommended only with GPU and sufficient data. |

### Hyperparameter Options

```text
--experiment-id       Unique name for the experiment run (default: resnet50_optimized)
--architecture        Model backbone: resnet50 (default) or efficientnet_b0
--epochs              Number of training epochs (default: 8)
--batch-size          Mini-batch size (default: 16)
--backbone-lr         Learning rate for backbone layers (default: 2e-5)
--head-lr             Learning rate for classification head (default: 2e-4)
--dropout             Dropout rate before classification layer (default: 0.2)
--label-smoothing     Cross-entropy label smoothing factor (default: 0.05)
--scheduler           Learning rate scheduler: plateau (default), cosine, or none
--seed                Global random seed for reproducibility (default: 42)
--set-as-production   Copies best checkpoint to backend/models/acne_resnet50.pth
```

---

## 4. Standalone Evaluation Command

To evaluate any existing checkpoint against any split and generate confusion matrices and classification reports:

```powershell
cd Skinora\backend
..\venv\Scripts\python.exe -m Skinora.backend.training.evaluate_acne_model --checkpoint models\acne_resnet50.pth --split test --output-dir experiments\baseline_resnet50
```

---

## 5. Artifact Outputs

Each training run automatically produces:
1. `checkpoint.pth`: Best model selected based on **Validation Macro F1**.
2. `training_history.json`: Epoch-by-epoch loss, accuracy, and macro F1 curves.
3. `evaluation_test.json`: Full metrics on the held-out test split.
4. `classification_report_test.csv`: Machine-readable per-class performance.
5. `confusion_matrix.png` & `confusion_matrix_normalized.png`: Visual heatmap plots.
6. `training_curves.png`: Loss and performance trajectories over epochs.
7. `manifest.json`: Complete experiment provenance and configuration metadata.

