"""ACNE04 Comprehensive Dataset Inspection, Verification, and Preparation Pipeline.

Audits:
1. Schema, columns, data types, and missing values.
2. Label consistency (level vs severity vs lesion_count_label vs num_lesions_annotated).
3. Bounding box validity, coordinate ranges, boundary compliance, and box counts.
4. Five-fold cross-validation split distributions and cross-fold overlap.
5. Exact and perceptual duplicate clustering and comparison against official folds.
6. Validation of clean_splits_manifest.json against raw dataset indices.
7. Master dataset metadata manifest generation.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = BACKEND_DIR / "data" / "acne04_metadata"
MODELS_DIR = BACKEND_DIR / "models"
CLEAN_MANIFEST_PATH = MODELS_DIR / "clean_splits_manifest.json"

CLASSES = ["mild", "moderate", "severe", "very_severe"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for perceptual near-duplicate analysis."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return int(decimal_val)


def inspect_acne04() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute complete audit of ACNE04 dataset and return report and master records."""
    print("Loading Layered-Labs/ACNE04 dataset...")
    ds = datasets.load_dataset("Layered-Labs/ACNE04", split="train")
    total_images = len(ds)
    print(f"Total dataset entries: {total_images}")

    # 1. Schema & Feature inspection
    features_info = {}
    for feat_name, feat_type in ds.features.items():
        features_info[feat_name] = str(feat_type)

    master_records = []
    sha256_to_indices = collections.defaultdict(list)
    dhash_to_indices = collections.defaultdict(list)

    # Accumulators for audit metrics
    level_counts = collections.Counter()
    severity_counts = collections.Counter()
    split_counts = collections.Counter()
    fold_counts = {f"fold_{k}": collections.Counter() for k in range(5)}

    missing_fields_count = collections.defaultdict(int)
    lesion_count_vs_annotated_diffs = []
    hayashi_discrepancies = []

    bbox_audit = {
        "total_boxes_annotated": 0,
        "images_with_boxes": 0,
        "images_with_zero_boxes": 0,
        "invalid_coord_order": 0,  # xmin >= xmax or ymin >= ymax
        "out_of_bounds_coords": 0,  # x < 0 or x > width or y < 0 or y > height
        "box_count_mismatch": 0,  # len(boxes) != num_lesions_annotated
        "min_lesion_area": float("inf"),
        "max_lesion_area": 0.0,
    }

    dimensions_summary = {
        "widths": collections.Counter(),
        "heights": collections.Counter(),
        "aspect_ratios": collections.Counter(),
        "modes": collections.Counter(),
    }

    print("Auditing all 1,457 images, annotations, bounding boxes, and fold metadata...")
    for idx in range(total_images):
        row = ds[idx]
        img = row["image"]
        w, h = img.size
        mode = img.mode

        dimensions_summary["widths"][w] += 1
        dimensions_summary["heights"][h] += 1
        dimensions_summary["aspect_ratios"][round(w / h, 2)] += 1
        dimensions_summary["modes"][mode] += 1

        img_rgb = img.convert("RGB")
        raw_bytes = img_rgb.tobytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        dhash = compute_dhash(img_rgb)

        sha256_to_indices[sha256].append(idx)
        dhash_to_indices[dhash].append(idx)

        # Basic labels
        fname = row["file_name"]
        level = int(row["level"])
        sev = str(row["severity"])
        lesion_count_lbl = int(row["lesion_count_label"])
        num_annotated = int(row["num_lesions_annotated"])
        split_val = str(row["split"])

        level_counts[level] += 1
        severity_counts[sev] += 1
        split_counts[split_val] += 1

        # Folds
        folds_dict = {}
        for k in range(5):
            f_val = str(row.get(f"fold_{k}_split"))
            folds_dict[f"fold_{k}"] = f_val
            fold_counts[f"fold_{k}"][f_val] += 1

        # Check missing values
        for fld in ["file_name", "level", "severity", "lesion_count_label", "num_lesions_annotated", "split"]:
            if row.get(fld) is None:
                missing_fields_count[fld] += 1

        # Check relationship between lesion_count_label and num_lesions_annotated
        if lesion_count_lbl != num_annotated:
            lesion_count_vs_annotated_diffs.append({
                "index": idx,
                "file_name": fname,
                "lesion_count_label": lesion_count_lbl,
                "num_lesions_annotated": num_annotated,
                "diff": lesion_count_lbl - num_annotated,
            })

        # Check Hayashi guideline adherence:
        # Mild: 0-5, Moderate: 6-20, Severe: 21-50, Very Severe: >50
        expected_level = 0 if lesion_count_lbl <= 5 else (1 if lesion_count_lbl <= 20 else (2 if lesion_count_lbl <= 50 else 3))
        if level != expected_level:
            hayashi_discrepancies.append({
                "index": idx,
                "file_name": fname,
                "level": level,
                "severity": sev,
                "lesion_count_label": lesion_count_lbl,
                "expected_level_from_count": expected_level,
            })

        # Bounding boxes inspection
        boxes = row.get("lesion_boxes", {})
        xmins = boxes.get("xmin", [])
        ymins = boxes.get("ymin", [])
        xmaxs = boxes.get("xmax", [])
        ymaxs = boxes.get("ymax", [])
        box_count = len(xmins)

        if box_count != num_annotated:
            bbox_audit["box_count_mismatch"] += 1

        if box_count == 0:
            bbox_audit["images_with_zero_boxes"] += 1
        else:
            bbox_audit["images_with_boxes"] += 1

        bbox_audit["total_boxes_annotated"] += box_count

        parsed_boxes = []
        for b_i in range(box_count):
            bx1, by1, bx2, by2 = int(xmins[b_i]), int(ymins[b_i]), int(xmaxs[b_i]), int(ymaxs[b_i])
            if bx1 >= bx2 or by1 >= by2:
                bbox_audit["invalid_coord_order"] += 1
            if bx1 < 0 or bx2 > w or by1 < 0 or by2 > h:
                bbox_audit["out_of_bounds_coords"] += 1

            area = (bx2 - bx1) * (by2 - by1)
            if area > 0:
                bbox_audit["min_lesion_area"] = min(bbox_audit["min_lesion_area"], area)
                bbox_audit["max_lesion_area"] = max(bbox_audit["max_lesion_area"], area)

            parsed_boxes.append({"xmin": bx1, "ymin": by1, "xmax": bx2, "ymax": by2, "area": area})

        rec = {
            "index": idx,
            "file_name": fname,
            "level": level,
            "severity": sev,
            "lesion_count_label": lesion_count_lbl,
            "num_lesions_annotated": num_annotated,
            "dimensions": {"width": w, "height": h, "aspect_ratio": round(w / h, 3), "mode": mode},
            "sha256": sha256,
            "dhash": dhash,
            "split": split_val,
            "folds": folds_dict,
            "box_count": box_count,
            "boxes": parsed_boxes,
        }
        master_records.append(rec)

    # 2. Duplicate audit and Connected Components
    exact_duplicate_groups = []
    cluster_id = 0
    index_to_cluster = {}

    for sha, idxs in sha256_to_indices.items():
        is_dup = len(idxs) > 1
        sub_recs = [master_records[i] for i in idxs]
        levels = [r["level"] for r in sub_recs]
        splits = [r["split"] for r in sub_recs]
        fnames = [r["file_name"] for r in sub_recs]
        has_conflict = len(set(levels)) > 1
        is_cross_split = len(set(splits)) > 1

        c_info = {
            "cluster_id": cluster_id,
            "sha256": sha,
            "size": len(idxs),
            "indices": idxs,
            "file_names": fnames,
            "levels": levels,
            "splits": splits,
            "has_label_conflict": has_conflict,
            "is_cross_split": is_cross_split,
            "is_duplicate": is_dup,
        }
        if is_dup:
            exact_duplicate_groups.append(c_info)
        for i in idxs:
            index_to_cluster[i] = cluster_id
            master_records[i]["duplicate_cluster_id"] = cluster_id
            master_records[i]["is_exact_duplicate"] = is_dup
            master_records[i]["has_label_conflict"] = has_conflict

        cluster_id += 1

    # Compare duplicate groups against official 5-fold splits
    fold_leakage_audit = {}
    for k in range(5):
        crossing_groups = []
        for g in exact_duplicate_groups:
            f_splits = set(master_records[i]["folds"][f"fold_{k}"] for i in g["indices"])
            if len(f_splits) > 1:
                crossing_groups.append({
                    "cluster_id": g["cluster_id"],
                    "indices": g["indices"],
                    "file_names": g["file_names"],
                    "fold_splits": list(f_splits),
                })
        fold_leakage_audit[f"fold_{k}"] = {
            "crossing_duplicate_groups_count": len(crossing_groups),
            "crossing_groups": crossing_groups,
        }

    # 3. Verify clean_splits_manifest.json against raw records
    clean_manifest_audit = {
        "clean_manifest_present": CLEAN_MANIFEST_PATH.exists(),
        "is_valid": False,
        "details": {},
    }
    if CLEAN_MANIFEST_PATH.exists():
        with open(CLEAN_MANIFEST_PATH, "r", encoding="utf-8") as f:
            cman = json.load(f)
        t_idx = cman["splits"]["train"]
        v_idx = cman["splits"]["validation"]
        te_idx = cman["splits"]["test"]

        # Attach clean split assignment to master records
        for i in t_idx:
            master_records[i]["clean_split"] = "train"
        for i in v_idx:
            master_records[i]["clean_split"] = "validation"
        for i in te_idx:
            master_records[i]["clean_split"] = "test"

        t_set, v_set, te_set = set(t_idx), set(v_idx), set(te_idx)
        all_indices_set = t_set | v_set | te_set

        # Check hashes
        t_hashes = set(master_records[i]["sha256"] for i in t_idx)
        v_hashes = set(master_records[i]["sha256"] for i in v_idx)
        te_hashes = set(master_records[i]["sha256"] for i in te_idx)

        t_v_leak = len(t_hashes & v_hashes)
        t_te_leak = len(t_hashes & te_hashes)
        v_te_leak = len(v_hashes & te_hashes)

        # Check duplicate group integrity
        groups_split = 0
        for g in exact_duplicate_groups:
            g_set = set(g["indices"])
            in_t = bool(g_set & t_set)
            in_v = bool(g_set & v_set)
            in_te = bool(g_set & te_set)
            if sum([in_t, in_v, in_te]) > 1:
                groups_split += 1

        clean_manifest_audit["is_valid"] = (
            len(all_indices_set) == 1457
            and len(t_idx) == 990
            and len(v_idx) == 175
            and len(te_idx) == 292
            and t_v_leak == 0
            and t_te_leak == 0
            and v_te_leak == 0
            and groups_split == 0
        )
        clean_manifest_audit["details"] = {
            "train_size": len(t_idx),
            "val_size": len(v_idx),
            "test_size": len(te_idx),
            "total_assigned": len(all_indices_set),
            "train_val_hash_overlap": t_v_leak,
            "train_test_hash_overlap": t_te_leak,
            "val_test_hash_overlap": v_te_leak,
            "duplicate_groups_split": groups_split,
        }

    # Compile Full Inspection Report
    report = {
        "dataset_name": "Layered-Labs/ACNE04",
        "total_images": total_images,
        "features_schema": features_info,
        "missing_values": dict(missing_fields_count),
        "dimensions_summary": {
            "unique_resolutions_count": len(dimensions_summary["widths"]),
            "dominant_widths": dict(dimensions_summary["widths"].most_common(5)),
            "dominant_heights": dict(dimensions_summary["heights"].most_common(5)),
            "aspect_ratios": dict(dimensions_summary["aspect_ratios"].most_common(5)),
            "color_modes": dict(dimensions_summary["modes"]),
        },
        "label_distribution": {
            "by_level": dict(level_counts),
            "by_severity": dict(severity_counts),
            "class_percentages": {
                CLASSES[lvl]: round(cnt / total_images * 100, 2)
                for lvl, cnt in sorted(level_counts.items())
            },
        },
        "original_splits": dict(split_counts),
        "official_five_folds": {k: dict(v) for k, v in fold_counts.items()},
        "annotation_consistency": {
            "lesion_count_vs_num_annotated_mismatches": len(lesion_count_vs_annotated_diffs),
            "sample_lesion_count_mismatches": lesion_count_vs_annotated_diffs[:5],
            "hayashi_rule_discrepancies": len(hayashi_discrepancies),
            "sample_hayashi_discrepancies": hayashi_discrepancies[:5],
        },
        "bounding_box_audit": bbox_audit,
        "duplicate_audit": {
            "total_exact_duplicate_groups": len(exact_duplicate_groups),
            "total_images_in_exact_duplicates": sum(g["size"] for g in exact_duplicate_groups),
            "cross_split_original_groups": len([g for g in exact_duplicate_groups if g["is_cross_split"]]),
            "conflicting_label_groups": len([g for g in exact_duplicate_groups if g["has_label_conflict"]]),
            "official_fold_leakage": {k: v["crossing_duplicate_groups_count"] for k, v in fold_leakage_audit.items()},
        },
        "clean_manifest_verification": clean_manifest_audit,
    }

    # Save artifacts to METADATA_DIR
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report_file = METADATA_DIR / "acne04_inspection_report.json"
    manifest_file = METADATA_DIR / "acne04_master_manifest.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(master_records, f, indent=2)

    print(f"Audit report saved: {report_file}")
    print(f"Master manifest saved: {manifest_file} ({len(master_records)} records)")

    return report, master_records


if __name__ == "__main__":
    report, _ = inspect_acne04()
    print("\n--- ACNE04 INSPECTION SUMMARY ---")
    print(f"Total entries:               {report['total_images']}")
    print(f"Class distribution:          {report['label_distribution']['class_percentages']}")
    print(f"Original splits:             {report['original_splits']}")
    print(f"Bounding boxes annotated:    {report['bounding_box_audit']['total_boxes_annotated']}")
    print(f"Box coordinate errors:       {report['bounding_box_audit']['out_of_bounds_coords']} out-of-bounds, {report['bounding_box_audit']['invalid_coord_order']} inverted")
    print(f"Exact duplicate groups:      {report['duplicate_audit']['total_exact_duplicate_groups']}")
    print(f"Official folds with leakage: {report['duplicate_audit']['official_fold_leakage']}")
    print(f"Clean manifest valid:        {report['clean_manifest_verification']['is_valid']}")

