"""Unit tests for dataset integrity, duplicate group isolation, and split leakage verification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "clean_splits_manifest.json"
CACHE_PATH = BACKEND_DIR / "training" / ".cache" / "acne04_metadata.json"


def test_clean_splits_manifest_structure():
    assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    train_idx = splits["train"]
    val_idx = splits["validation"]
    test_idx = splits["test"]

    assert len(train_idx) == 990, f"Expected 990 train, got {len(train_idx)}"
    assert len(val_idx) == 175, f"Expected 175 val, got {len(val_idx)}"
    assert len(test_idx) == 292, f"Expected 292 test, got {len(test_idx)}"
    assert len(train_idx) + len(val_idx) + len(test_idx) == 1457

    # Disjoint check
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    assert len(s_train & s_val) == 0, "Index overlap between train and val!"
    assert len(s_train & s_test) == 0, "Index overlap between train and test!"
    assert len(s_val & s_test) == 0, "Index overlap between val and test!"


def test_zero_hash_leakage():
    assert CACHE_PATH.exists(), f"Metadata cache missing at {CACHE_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    splits = manifest["splits"]
    train_hashes = set(records[i]["sha256"] for i in splits["train"])
    val_hashes = set(records[i]["sha256"] for i in splits["validation"])
    test_hashes = set(records[i]["sha256"] for i in splits["test"])

    assert len(train_hashes & val_hashes) == 0, "Hash leakage between train and validation!"
    assert len(train_hashes & test_hashes) == 0, "Hash leakage between train and test!"
    assert len(val_hashes & test_hashes) == 0, "Hash leakage between validation and test!"


def test_duplicate_clusters_isolated():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    splits = manifest["splits"]
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    # Group by sha256
    import collections
    sha_map = collections.defaultdict(list)
    for r in records:
        sha_map[r["sha256"]].append(r["index"])

    duplicate_groups = [idxs for idxs in sha_map.values() if len(idxs) > 1]
    assert len(duplicate_groups) == 38, f"Expected 38 duplicate groups, found {len(duplicate_groups)}"

    for group in duplicate_groups:
        g_set = set(group)
        in_train = bool(g_set & train_set)
        in_val = bool(g_set & val_set)
        in_test = bool(g_set & test_set)
        partitions = sum([in_train, in_val, in_test])
        assert partitions == 1, f"Duplicate group {group} split across {partitions} partitions!"


def test_contradictory_labels_quarantined_to_train():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    splits = manifest["splits"]
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    import collections
    sha_map = collections.defaultdict(list)
    for r in records:
        sha_map[r["sha256"]].append(r["index"])

    conflict_groups = []
    for idxs in sha_map.values():
        if len(idxs) > 1:
            levels = set(records[i]["level"] for i in idxs)
            if len(levels) > 1:
                conflict_groups.append(idxs)

    assert len(conflict_groups) == 8, f"Expected 8 contradictory groups, found {len(conflict_groups)}"

    for group in conflict_groups:
        g_set = set(group)
        # All contradictory samples must be safely quarantined inside train
        assert g_set.issubset(train_set), f"Conflict group {group} leaked into val or test!"
        assert len(g_set & val_set) == 0, f"Conflict group {group} found in validation!"
        assert len(g_set & test_set) == 0, f"Conflict group {group} found in test!"


def test_production_checkpoint_preserved():
    prod_ckpt = MODELS_DIR / "acne_resnet50.pth"
    assert prod_ckpt.exists(), f"Production checkpoint missing at {prod_ckpt}"
    assert prod_ckpt.stat().st_size > 90_000_000, "Production checkpoint size unexpectedly small"


def test_class_id_mapping_stability():
    try:
        from training.config import CLASS_TO_ID, CLASSES
    except ImportError:
        try:
            from backend.training.config import CLASS_TO_ID, CLASSES
        except ImportError:
            from Skinora.backend.training.config import CLASS_TO_ID, CLASSES
    assert CLASSES == ["mild", "moderate", "severe", "very_severe"]
    assert CLASS_TO_ID == {"mild": 0, "moderate": 1, "severe": 2, "very_severe": 3}

