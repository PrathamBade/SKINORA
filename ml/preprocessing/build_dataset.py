"""
SKINORA ML — Dataset Builder

Reads metadata.jsonl from all_1024/ and produces acne_dataset.csv,
then splits it into stratified train / val / test CSVs.

Usage:
    cd SKINORA/ml
    python -m preprocessing.build_dataset

Outputs (in ml/datasets/processed/):
    acne_dataset.csv   — full dataset (filename, acne_level, image_path)
    train.csv
    val.csv
    test.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running as a standalone script or as part of the ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    DATASET_CSV,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    TEST_CSV,
    TRAIN_CSV,
    TRAIN_RATIO,
    VAL_CSV,
)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

_LEVEL_FROM_NAME = re.compile(r"levle(\d+)_", re.IGNORECASE)
_LEVEL_FROM_PROMPT = re.compile(r"acne(\d+)", re.IGNORECASE)


def _parse_level(filename: str, prompt: str) -> int | None:
    """Extract the integer acne severity level (0–3) from filename or prompt.

    Filename convention: levle{level}_{id}.jpg
    Prompt convention:   "photo of a person with acne{level}"

    Returns None if no level can be determined.
    """
    m = _LEVEL_FROM_NAME.search(filename)
    if m:
        return int(m.group(1))
    m = _LEVEL_FROM_PROMPT.search(prompt)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_dataset() -> pd.DataFrame:
    """Parse metadata.jsonl → acne_dataset.csv.

    Returns the DataFrame with columns: filename, acne_level, image_path
    """
    metadata_path = RAW_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found at: {metadata_path}")

    records = []
    skipped = 0

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            filename: str = entry["file_name"]
            prompt: str = entry.get("prompt", "")

            level = _parse_level(filename, prompt)
            if level is None:
                print(f"  WARNING: Could not determine level for '{filename}' — skipping.")
                skipped += 1
                continue

            if level not in (0, 1, 2, 3):
                print(f"  WARNING: Unexpected level {level} for '{filename}' — skipping.")
                skipped += 1
                continue

            image_path = RAW_DIR / filename
            if not image_path.exists():
                print(f"  WARNING: Image file not found: {image_path} — skipping.")
                skipped += 1
                continue

            records.append({
                "filename": filename,
                "acne_level": level,
                "image_path": str(image_path),
            })

    df = pd.DataFrame(records)
    print(f"\nDataset built: {len(df)} images, {skipped} skipped.")
    print("\nClass distribution:")
    print(df["acne_level"].value_counts().sort_index().to_string())

    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 split.

    Returns: (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=df["acne_level"],
        random_state=RANDOM_SEED,
    )
    # Split the remaining 30% into val (50%) and test (50%) → 15% each
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["acne_level"],
        random_state=RANDOM_SEED,
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train_df):>5} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df):>5} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df):>5} ({len(test_df)/len(df)*100:.1f}%)")

    # Sanity check: no overlap between splits
    train_files = set(train_df["filename"])
    val_files = set(val_df["filename"])
    test_files = set(test_df["filename"])
    assert len(train_files & val_files) == 0, "Overlap between train and val!"
    assert len(train_files & test_files) == 0, "Overlap between train and test!"
    assert len(val_files & test_files) == 0, "Overlap between val and test!"
    print("\nSplit validation: no overlap between sets. [OK]")

    return train_df, val_df, test_df


def save_csvs(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist all four CSVs to the processed directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATASET_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\nCSVs saved to: {PROCESSED_DIR}")
    for csv_path in [DATASET_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV]:
        print(f"  {csv_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SKINORA — Dataset Builder")
    print("=" * 60)
    print(f"\nSource: {RAW_DIR}")

    df = build_dataset()
    train_df, val_df, test_df = split_dataset(df)
    save_csvs(df, train_df, val_df, test_df)

    print("\nDone. Dataset is ready for training.")
