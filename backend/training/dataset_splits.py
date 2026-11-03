"""Leakage-resistant, group-aware dataset splitting for ACNE04 in Skinora."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MANIFEST_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "clean_splits_manifest.json"


def compute_dhash(image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for near-duplicate analysis."""
    resized = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return int(decimal_val)


def extract_and_cache_dataset_metadata(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Extract SHA256 hashes and metadata from ACNE04, caching results locally."""
    cache_file = CACHE_DIR / "acne04_metadata.json"
    if not force_refresh and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    import datasets

    print("Extracting ACNE04 metadata and computing hashes...")
    ds = datasets.load_dataset("Layered-Labs/ACNE04", split="train")
    records = []

    for idx in range(len(ds)):
        row = ds[idx]
        img = row["image"].convert("RGB")
        img_bytes = img.tobytes()
        sha256 = hashlib.sha256(img_bytes).hexdigest()
        dhash = compute_dhash(img)

        rec = {
            "index": idx,
            "file_name": row["file_name"],
            "level": int(row["level"]),
            "severity": row["severity"],
            "original_split": str(row["split"]),
            "lesion_count": int(row["lesion_count_label"]),
            "num_annotated": int(row["num_lesions_annotated"]),
            "sha256": sha256,
            "dhash": dhash,
            "image_size": list(img.size),
        }
        records.append(rec)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return records


def build_duplicate_clusters(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Identify exact duplicate connected components and map each image to cluster ID."""
    hash_to_indices = collections.defaultdict(list)
    for r in records:
        hash_to_indices[r["sha256"]].append(r["index"])

    clusters = []
    index_to_cluster = {}

    cluster_id = 0
    for sha, indices in hash_to_indices.items():
        sub_records = [records[i] for i in indices]
        levels = [r["level"] for r in sub_records]
        splits = [r["original_split"] for r in sub_records]
        files = [r["file_name"] for r in sub_records]
        has_conflict = len(set(levels)) > 1
        is_cross_split = len(set(splits)) > 1

        primary_level = collections.Counter(levels).most_common(1)[0][0]

        cluster_info = {
            "cluster_id": cluster_id,
            "sha256": sha,
            "indices": indices,
            "file_names": files,
            "size": len(indices),
            "levels": levels,
            "primary_level": primary_level,
            "original_splits": list(set(splits)),
            "has_label_conflict": has_conflict,
            "is_cross_split": is_cross_split,
            "is_duplicate": len(indices) > 1,
        }
        clusters.append(cluster_info)
        for i in indices:
            index_to_cluster[i] = cluster_id
        cluster_id += 1

    return clusters, index_to_cluster


def generate_clean_split_manifest(
    records: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    seed: int = 42,
    target_train_size: int = 990,
    target_val_size: int = 175,
    target_test_size: int = 292,
) -> Dict[str, Any]:
    """Generate reproducible, group-aware, leakage-free split manifest."""
    rng = np.random.RandomState(seed)

    # Policy:
    # 1. Any cluster with label conflict MUST go to train (avoid corrupted test/val ground truth)
    # 2. Clusters with duplicates (size > 1) MUST NOT be split across partitions
    # 3. Target counts: train ~990, val ~175, test ~292
    # 4. Stratification: keep class distributions across train, val, test balanced

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    # Separate clusters into buckets
    conflict_clusters = [c for c in clusters if c["has_label_conflict"]]
    for c in conflict_clusters:
        train_indices.extend(c["indices"])

    remaining_clusters = [c for c in clusters if not c["has_label_conflict"]]

    # Further separate into duplicate clusters (size > 1) and singletons (size == 1)
    dup_clusters = [c for c in remaining_clusters if c["is_duplicate"]]
    singleton_clusters = [c for c in remaining_clusters if not c["is_duplicate"]]

    # For duplicate clusters without conflict:
    # If they were originally in test only, prefer test/val/train while keeping cluster intact
    # Shuffle duplicate clusters deterministically
    dup_perm = rng.permutation(len(dup_clusters))
    dup_clusters = [dup_clusters[i] for i in dup_perm]

    # Assign duplicate clusters by their primary level into splits
    for c in dup_clusters:
        # Check current sizes
        if len(test_indices) + c["size"] <= target_test_size and "test" in c["original_splits"]:
            test_indices.extend(c["indices"])
        elif len(val_indices) + c["size"] <= target_val_size:
            val_indices.extend(c["indices"])
        else:
            train_indices.extend(c["indices"])

    # Now assign singletons by class stratification
    singletons_by_class: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in singleton_clusters:
        singletons_by_class[c["primary_level"]].append(c)

    for lvl in range(4):
        s_list = singletons_by_class[lvl]
        perm = rng.permutation(len(s_list))
        s_list = [s_list[i] for i in perm]
        singletons_by_class[lvl] = s_list

    # Calculate remaining slots needed
    # Compute class targets for val and test based on overall class proportions
    total_samples = len(records)
    class_totals = np.bincount([r["level"] for r in records], minlength=4)
    class_ratios = class_totals / total_samples

    val_class_targets = np.round(class_ratios * target_val_size).astype(int)
    test_class_targets = np.round(class_ratios * target_test_size).astype(int)

    # Adjust rounding if needed
    val_class_targets[1] += (target_val_size - int(val_class_targets.sum()))
    test_class_targets[1] += (target_test_size - int(test_class_targets.sum()))

    # Fill test slots from singletons first
    for lvl in range(4):
        needed_test = test_class_targets[lvl] - sum(1 for i in test_indices if records[i]["level"] == lvl)
        needed_test = max(0, needed_test)
        
        # Prefer singletons that were originally in test
        orig_test_s = [c for c in singletons_by_class[lvl] if c["original_splits"] == ["test"]]
        orig_train_s = [c for c in singletons_by_class[lvl] if c["original_splits"] != ["test"]]
        
        candidates = orig_test_s + orig_train_s
        chosen_test = candidates[:needed_test]
        remaining = candidates[needed_test:]
        
        for c in chosen_test:
            test_indices.extend(c["indices"])
        singletons_by_class[lvl] = remaining

    # Fill val slots from remaining singletons
    for lvl in range(4):
        needed_val = val_class_targets[lvl] - sum(1 for i in val_indices if records[i]["level"] == lvl)
        needed_val = max(0, needed_val)
        
        candidates = singletons_by_class[lvl]
        chosen_val = candidates[:needed_val]
        remaining = candidates[needed_val:]
        
        for c in chosen_val:
            val_indices.extend(c["indices"])
        singletons_by_class[lvl] = remaining

    # Everything remaining goes to train
    for lvl in range(4):
        for c in singletons_by_class[lvl]:
            train_indices.extend(c["indices"])

    # Sort indices
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # Build manifest
    manifest = {
        "manifest_version": "1.0",
        "dataset_name": "Layered-Labs/ACNE04",
        "description": "Leakage-free, group-stratified split partition for Skinora",
        "seed": seed,
        "total_samples": total_samples,
        "split_counts": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "class_distributions": {
            "train": [int(x) for x in np.bincount([records[i]["level"] for i in train_indices], minlength=4)],
            "validation": [int(x) for x in np.bincount([records[i]["level"] for i in val_indices], minlength=4)],
            "test": [int(x) for x in np.bincount([records[i]["level"] for i in test_indices], minlength=4)],
        },
        "integrity_verification": {
            "duplicate_groups_total": len([c for c in clusters if c["is_duplicate"]]),
            "label_conflicts_total": len(conflict_clusters),
            "label_conflicts_in_test": sum(1 for i in test_indices if any(i in c["indices"] for c in conflict_clusters)),
            "label_conflicts_in_val": sum(1 for i in val_indices if any(i in c["indices"] for c in conflict_clusters)),
            "label_conflicts_in_train": sum(1 for i in train_indices if any(i in c["indices"] for c in conflict_clusters)),
            "cross_split_leakage_detected": False,
        },
        "splits": {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        },
    }

    # Verify zero leakage
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    assert len(train_set & val_set) == 0, "Leakage between train and val!"
    assert len(train_set & test_set) == 0, "Leakage between train and test!"
    assert len(val_set & test_set) == 0, "Leakage between val and test!"

    # Verify no cluster split across partitions
    for c in clusters:
        c_set = set(c["indices"])
        in_train = bool(c_set & train_set)
        in_val = bool(c_set & val_set)
        in_test = bool(c_set & test_set)
        num_partitions = sum([in_train, in_val, in_test])
        if num_partitions > 1:
            manifest["integrity_verification"]["cross_split_leakage_detected"] = True
            raise ValueError(f"Cluster {c['cluster_id']} split across {num_partitions} partitions!")

    return manifest


def verify_split_manifest(manifest: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute rigorous verification of split manifest against raw records."""
    train_idx = manifest["splits"]["train"]
    val_idx = manifest["splits"]["validation"]
    test_idx = manifest["splits"]["test"]

    train_hashes = set(records[i]["sha256"] for i in train_idx)
    val_hashes = set(records[i]["sha256"] for i in val_idx)
    test_hashes = set(records[i]["sha256"] for i in test_idx)

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    return {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "total": len(train_idx) + len(val_idx) + len(test_idx),
        "train_val_overlap_hashes": len(train_val_overlap),
        "train_test_overlap_hashes": len(train_test_overlap),
        "val_test_overlap_hashes": len(val_test_overlap),
        "is_leakage_free": (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        ),
    }


def save_manifest(manifest: Dict[str, Any], output_path: Path = MANIFEST_DEFAULT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved clean split manifest to {output_path}")


def load_manifest(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    target = path or MANIFEST_DEFAULT_PATH
    if not target.exists():
        return None
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    records = extract_and_cache_dataset_metadata()
    clusters, idx_to_c = build_duplicate_clusters(records)
    print(f"Total clusters: {len(clusters)}")
    
    manifest = generate_clean_split_manifest(records, clusters, seed=42)
    verification = verify_split_manifest(manifest, records)
    print("\n--- SPLIT MANIFEST VERIFICATION ---")
    print(json.dumps(verification, indent=2))
    print(f"\nSplit counts: {manifest['split_counts']}")
    print(f"Class distributions: {json.dumps(manifest['class_distributions'], indent=2)}")
    
    save_manifest(manifest)

