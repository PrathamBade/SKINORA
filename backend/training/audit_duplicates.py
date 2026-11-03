"""Dataset Integrity and Duplicate Audit Script for ACNE04."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import datasets
import numpy as np
from PIL import Image


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for perceptual near-duplicate detection."""
    # Resize to (hash_size + 1, hash_size), convert to grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int32)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to integer bitmask
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def main():
    print("Loading ACNE04 dataset...")
    ds = datasets.load_dataset("Layered-Labs/ACNE04", split="train")
    total = len(ds)
    print(f"Total rows: {total}")

    records = []
    sha256_to_indices = collections.defaultdict(list)

    print("Computing hashes and extracting metadata...")
    for idx in range(total):
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
            "split": str(row["split"]),
            "lesion_count": int(row["lesion_count_label"]),
            "num_annotated": int(row["num_lesions_annotated"]),
            "fold_0": row.get("fold_0_split"),
            "fold_1": row.get("fold_1_split"),
            "fold_2": row.get("fold_2_split"),
            "fold_3": row.get("fold_3_split"),
            "fold_4": row.get("fold_4_split"),
            "sha256": sha256,
            "dhash": dhash,
            "size": img.size,
        }
        records.append(rec)
        sha256_to_indices[sha256].append(idx)

    # 1. Exact Duplicates
    exact_duplicate_groups = [
        idxs for idxs in sha256_to_indices.values() if len(idxs) > 1
    ]
    print(f"\n--- EXACT DUPLICATE AUDIT ---")
    print(f"Total exact duplicate groups: {len(exact_duplicate_groups)}")
    total_exact_dup_images = sum(len(g) for g in exact_duplicate_groups)
    print(f"Total images involved in exact duplicates: {total_exact_dup_images}")

    # Inspect each exact duplicate group
    cross_split_exact = []
    label_conflict_exact = []

    for g_idx, group in enumerate(exact_duplicate_groups):
        group_recs = [records[i] for i in group]
        splits = set(r["split"] for r in group_recs)
        levels = set(r["level"] for r in group_recs)
        file_names = [r["file_name"] for r in group_recs]
        
        is_cross = len(splits) > 1
        is_conflict = len(levels) > 1

        if is_cross:
            cross_split_exact.append((g_idx, group_recs))
        if is_conflict:
            label_conflict_exact.append((g_idx, group_recs))

    print(f"Exact groups crossing train/test splits: {len(cross_split_exact)}")
    print(f"Exact groups with conflicting severity labels: {len(label_conflict_exact)}")

    print("\nDetails of label-conflicting exact groups:")
    for g_idx, group_recs in label_conflict_exact:
        print(f"Group {g_idx}:")
        for r in group_recs:
            print(f"   idx={r['index']}, file={r['file_name']}, split={r['split']}, level={r['level']} ({r['severity']}), lesions={r['lesion_count']}")

    print("\nDetails of cross-split exact groups (train vs test):")
    for g_idx, group_recs in cross_split_exact:
        splits_str = ", ".join(f"{r['file_name']}({r['split']}, lvl={r['level']})" for r in group_recs)
        print(f"Group {g_idx}: {splits_str}")

    # Check official folds for exact duplicate groups
    print("\n--- OFFICIAL 5-FOLD CROSS-SPLIT BEHAVIOR ON EXACT DUPLICATES ---")
    for fold_k in range(5):
        fold_cross_count = 0
        for group in exact_duplicate_groups:
            group_folds = set(records[i][f"fold_{fold_k}"] for i in group)
            if len(group_folds) > 1:
                fold_cross_count += 1
        print(f"fold_{fold_k}_split has {fold_cross_count} exact duplicate groups crossing train/test")

    # 2. Near Duplicates via dHash
    print("\n--- NEAR-DUPLICATE AUDIT (dHash Hamming distance <= 2) ---")
    near_dup_pairs = []
    # Check all pairs that are not exact duplicates
    for i in range(total):
        for j in range(i + 1, total):
            if records[i]["sha256"] != records[j]["sha256"]:
                h_dist = hamming_distance(records[i]["dhash"], records[j]["dhash"])
                if h_dist <= 2:
                    near_dup_pairs.append((i, j, h_dist))

    print(f"Near-duplicate pairs (non-exact, dHash dist <= 2): {len(near_dup_pairs)}")
    for i, j, dist in near_dup_pairs[:20]:
        r1, r2 = records[i], records[j]
        print(f"Dist {dist}: idx {i} ({r1['file_name']}, split={r1['split']}, lvl={r1['level']}) <-> idx {j} ({r2['file_name']}, split={r2['split']}, lvl={r2['level']})")

    # 3. Filename patterns and potential patient IDs
    print("\n--- FILENAME ANALYSIS ---")
    # ACNE04 filenames usually have format like levleX_Y.jpg
    # Check if there are other filename patterns or prefix clues
    prefixes = collections.Counter()
    for r in records:
        parts = r["file_name"].split("_")
        prefixes[parts[0]] += 1
    print(f"Filename prefix counts: {dict(prefixes)}")


if __name__ == "__main__":
    main()

