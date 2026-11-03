"""Verification script for ACNE04 dataset-to-clean-manifest mapping integrity."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

# Ensure root paths are in sys.path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

BACKEND_DIR = Path(__file__).resolve().parents[1]
CLEAN_MANIFEST_PATH = BACKEND_DIR / "models" / "clean_splits_manifest.json"
MASTER_MANIFEST_PATH = BACKEND_DIR / "data" / "acne04_metadata" / "acne04_master_manifest.json"

DATASET_NAME = "Layered-Labs/ACNE04"
DATASET_REVISION = "e375d8dac98b8562c5ab462d5f00ac011e8a47db"
CLASSES = ["mild", "moderate", "severe", "very_severe"]


def verify_mapping():
    print(f"Loading ACNE04 from Hugging Face ({DATASET_NAME}, revision={DATASET_REVISION})...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train", revision=DATASET_REVISION)

    print(f"Loaded {len(ds)} rows from ACNE04 dataset.")
    assert len(ds) == 1457, f"Expected 1457 rows, got {len(ds)}"

    # 1. Clean split manifest verification
    print(f"Reading clean splits manifest from {CLEAN_MANIFEST_PATH}...")
    with open(CLEAN_MANIFEST_PATH, "r", encoding="utf-8") as f:
        clean_man = json.load(f)

    splits = clean_man["splits"]
    train_idx = splits["train"]
    val_idx = splits["validation"]
    test_idx = splits["test"]

    print(f"  Train:      {len(train_idx)} entries")
    print(f"  Validation: {len(val_idx)} entries")
    print(f"  Test:       {len(test_idx)} entries")
    total_assigned = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"  Total:      {total_assigned} entries")

    assert len(train_idx) == 990, f"Expected 990 train, got {len(train_idx)}"
    assert len(val_idx) == 175, f"Expected 175 val, got {len(val_idx)}"
    assert len(test_idx) == 292, f"Expected 292 test, got {len(test_idx)}"
    assert total_assigned == 1457, f"Total {total_assigned} != 1457"

    # Set disjointness & full coverage
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    assert len(s_train) == 990, "Duplicate entries in train split!"
    assert len(s_val) == 175, "Duplicate entries in validation split!"
    assert len(s_test) == 292, "Duplicate entries in test split!"

    overlap_train_val = s_train & s_val
    overlap_train_test = s_train & s_test
    overlap_val_test = s_val & s_test
    assert len(overlap_train_val) == 0, f"Overlap train-val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Overlap train-test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Overlap val-test: {overlap_val_test}"

    union_all = s_train | s_val | s_test
    assert union_all == set(range(1457)), "Splits do not cover exactly indices 0..1456!"

    # 2. Compare against master manifest
    print(f"Reading master metadata manifest from {MASTER_MANIFEST_PATH}...")
    with open(MASTER_MANIFEST_PATH, "r", encoding="utf-8") as f:
        master_man = json.load(f)

    assert len(master_man) == 1457, f"Master manifest has {len(master_man)} entries, expected 1457"

    # 3. Check every single row
    for i in range(1457):
        row = ds[i]
        m_row = master_man[i]

        assert m_row["index"] == i, f"Row {i} index mismatch in master manifest"
        assert row["file_name"] == m_row["file_name"], f"Row {i} file_name mismatch: ds={row['file_name']}, master={m_row['file_name']}"
        assert int(row["level"]) == int(m_row["level"]), f"Row {i} level mismatch: ds={row['level']}, master={m_row['level']}"

        # Severity string check
        expected_sev = CLASSES[int(row["level"])]
        assert row["severity"] == expected_sev, f"Row {i} severity mismatch: {row['severity']} vs {expected_sev}"

        # Partition membership check
        expected_partition = "train" if i in s_train else ("validation" if i in s_val else "test")
        assert m_row["clean_split"] == expected_partition, (
            f"Row {i} clean_split mismatch: master has {m_row['clean_split']}, expected {expected_partition}"
        )

    print("  -> Verified all 1,457 rows match between ACNE04 dataset, master manifest, and clean split manifest.")

    # 4. Class distributions check
    levels = np.array(ds["level"], dtype=np.int64)
    train_counts = np.bincount(levels[train_idx], minlength=4).tolist()
    val_counts = np.bincount(levels[val_idx], minlength=4).tolist()
    test_counts = np.bincount(levels[test_idx], minlength=4).tolist()

    print(f"Observed Train counts: {train_counts} (Manifest: {clean_man['class_distributions']['train']})")
    print(f"Observed Val counts:   {val_counts} (Manifest: {clean_man['class_distributions']['validation']})")
    print(f"Observed Test counts:  {test_counts} (Manifest: {clean_man['class_distributions']['test']})")

    assert train_counts == clean_man["class_distributions"]["train"], "Train class distribution mismatch!"
    assert val_counts == clean_man["class_distributions"]["validation"], "Validation class distribution mismatch!"
    assert test_counts == clean_man["class_distributions"]["test"], "Test class distribution mismatch!"

    # 5. Duplicate group isolation check
    # Collect SHA-256 hashes
    train_hashes = set(master_man[i]["sha256"] for i in train_idx)
    val_hashes = set(master_man[i]["sha256"] for i in val_idx)
    test_hashes = set(master_man[i]["sha256"] for i in test_idx)

    assert len(train_hashes & val_hashes) == 0, "Hash leakage between train and val!"
    assert len(train_hashes & test_hashes) == 0, "Hash leakage between train and test!"
    assert len(val_hashes & test_hashes) == 0, "Hash leakage between val and test!"

    # Contradictory label duplicate pairs check
    import collections
    sha_to_indices = collections.defaultdict(list)
    for r in master_man:
        sha_to_indices[r["sha256"]].append(r["index"])

    conflict_groups = []
    for sha, indices in sha_to_indices.items():
        if len(indices) > 1:
            lbls = set(master_man[idx]["level"] for idx in indices)
            if len(lbls) > 1:
                conflict_groups.append(indices)

    assert len(conflict_groups) == 8, f"Expected 8 contradictory groups, got {len(conflict_groups)}"
    for cg in conflict_groups:
        cg_set = set(cg)
        assert cg_set.issubset(s_train), f"Conflict group {cg} leaked outside train!"

    print("  -> Verified zero cross-split hash leakage and all 8 contradictory duplicate pairs quarantined in train.")
    print("\nSUCCESS: All dataset-to-manifest mapping checks passed with 100% integrity!")


if __name__ == "__main__":
    verify_mapping()

