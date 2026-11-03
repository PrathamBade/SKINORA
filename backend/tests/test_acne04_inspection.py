"""Unit tests for ACNE04 dataset inspection, bounding-box validity, label consistency, and split reproducibility."""
from __future__ import annotations

import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = BACKEND_DIR / "data" / "acne04_metadata"
REPORT_PATH = METADATA_DIR / "acne04_inspection_report.json"
MASTER_MANIFEST_PATH = METADATA_DIR / "acne04_master_manifest.json"
CLEAN_MANIFEST_PATH = BACKEND_DIR / "models" / "clean_splits_manifest.json"
PRODUCTION_CKPT_PATH = BACKEND_DIR / "models" / "acne_resnet50.pth"


def test_inspection_artifacts_exist():
    assert REPORT_PATH.exists(), f"Inspection report missing at {REPORT_PATH}"
    assert MASTER_MANIFEST_PATH.exists(), f"Master manifest missing at {MASTER_MANIFEST_PATH}"


def test_dataset_size_and_missing_values():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["total_images"] == 1457
    # Zero missing values across all columns
    assert len(report["missing_values"]) == 0


def test_label_consistency_and_hayashi_rules():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    labels = report["label_distribution"]["by_level"]
    assert labels["0"] == 513  # Mild
    assert labels["1"] == 633  # Moderate
    assert labels["2"] == 182  # Severe
    assert labels["3"] == 129  # Very Severe
    assert sum(labels.values()) == 1457

    # Hayashi rule violations must be zero
    hayashi_errs = report["annotation_consistency"]["hayashi_rule_discrepancies"]
    assert hayashi_errs == 0, f"Found {hayashi_errs} Hayashi rule violations!"


def test_bounding_box_validity():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    bbox = report["bounding_box_audit"]
    assert bbox["total_boxes_annotated"] == 18983
    assert bbox["images_with_boxes"] == 1457
    assert bbox["images_with_zero_boxes"] == 0
    assert bbox["invalid_coord_order"] == 0
    assert bbox["out_of_bounds_coords"] == 0
    assert bbox["box_count_mismatch"] == 0


def test_official_five_folds_counts():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    folds = report["official_five_folds"]
    assert len(folds) == 5

    for k in range(5):
        f_counts = folds[f"fold_{k}"]
        assert f_counts["train"] == 1165
        assert f_counts["test"] == 292
        assert f_counts["train"] + f_counts["test"] == 1457


def test_duplicate_groups_and_leakage():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    dup = report["duplicate_audit"]
    assert dup["total_exact_duplicate_groups"] == 38
    assert dup["total_images_in_exact_duplicates"] == 76
    assert dup["cross_split_original_groups"] == 12
    assert dup["conflicting_label_groups"] == 8

    # All official folds leak duplicates
    for k in range(5):
        assert dup["official_fold_leakage"][f"fold_{k}"] >= 11


def test_clean_splits_manifest_verification():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    clean_audit = report["clean_manifest_verification"]
    assert clean_audit["clean_manifest_present"] is True
    assert clean_audit["is_valid"] is True

    details = clean_audit["details"]
    assert details["train_size"] == 990
    assert details["val_size"] == 175
    assert details["test_size"] == 292
    assert details["total_assigned"] == 1457
    assert details["train_val_hash_overlap"] == 0
    assert details["train_test_hash_overlap"] == 0
    assert details["val_test_hash_overlap"] == 0
    assert details["duplicate_groups_split"] == 0


def test_production_checkpoint_preserved():
    assert PRODUCTION_CKPT_PATH.exists(), f"Production checkpoint missing at {PRODUCTION_CKPT_PATH}"
    size = PRODUCTION_CKPT_PATH.stat().st_size
    assert size > 90_000_000, f"Production checkpoint size unexpectedly small ({size} bytes)"

