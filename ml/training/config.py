"""
SKINORA ML — Training Hyperparameters & Paths

All configurable settings for the training pipeline live here.
Change values in this file to experiment; training scripts read from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & key directories
# ---------------------------------------------------------------------------

ML_ROOT = Path(__file__).resolve().parent.parent          # ml/
DATASET_ROOT = ML_ROOT / "datasets"
RAW_DIR = DATASET_ROOT / "raw" / "ACNE04" / "acne_1024" / "all_1024"
PROCESSED_DIR = DATASET_ROOT / "processed"
MODELS_DIR = ML_ROOT / "models"
RESULTS_DIR = ML_ROOT / "results"

# Processed CSV paths
DATASET_CSV = PROCESSED_DIR / "acne_dataset.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Number of acne severity classes (0, 1, 2, 3)
NUM_CLASSES = 4

# Class labels for reporting
CLASS_NAMES = ["Level 0 (Clear)", "Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"]

# Train / val / test split ratios  (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ARCH = "resnet18"        # torchvision model name
PRETRAINED = True              # use ImageNet pretrained weights
FREEZE_BACKBONE = True         # freeze all layers except the final classifier

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224               # Input size for the model
BATCH_SIZE = 32
NUM_EPOCHS = 15                # Enough for a frozen backbone on CPU
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

MODEL_SAVE_PATH = MODELS_DIR / "acne_resnet18.pth"
RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"
CONFUSION_MATRIX_IMG = RESULTS_DIR / "confusion_matrix.png"
TRAINING_CURVE_IMG = RESULTS_DIR / "training_curves.png"
