# ACNE04 Dataset Inspection and Verification Report

**Dataset Identifier**: `Layered-Labs/ACNE04`  
**Total Images**: 1,457  
**Storage & Format**: Hugging Face Arrow Cache (unmodified, preserved)  
**Inspection Date**: November 2026  
**Master Manifest**: [`backend/data/acne04_metadata/acne04_master_manifest.json`](./acne04_master_manifest.json)  
**Machine-Readable Audit**: [`backend/data/acne04_metadata/acne04_inspection_report.json`](./acne04_inspection_report.json)

---

## 1. Schema, Column Types & Missing Values

Every entry in the dataset contains 12 fields. An audit across all 1,457 rows confirmed **0 missing or null values**:

| Column Name | Feature Type | Non-Null Count | Description |
| :--- | :--- | :---: | :--- |
| `image` | `Image(mode=None, decode=True)` | 1,457 | Facial photograph (100% RGB mode) |
| `file_name` | `string` | 1,457 | Source file identifier (e.g. `levle0_0.jpg`) |
| `level` | `int32` | 1,457 | Ordinal severity index (0, 1, 2, 3) |
| `severity` | `string` | 1,457 | Categorical label (`mild`, `moderate`, `severe`, `very_severe`) |
| `lesion_count_label` | `int32` | 1,457 | Clinical lesion count used for Hayashi grading |
| `num_lesions_annotated`| `int32` | 1,457 | Total count of annotated bounding boxes |
| `lesion_boxes` | `Dict[List[int32]]` | 1,457 | Dictionary containing `xmin`, `ymin`, `xmax`, `ymax` |
| `split` | `string` | 1,457 | Original binary partition (`train`: 1,165, `test`: 292) |
| `fold_0_split` | `string` | 1,457 | Official Fold 0 partition (`train`: 1,165, `test`: 292) |
| `fold_1_split` | `string` | 1,457 | Official Fold 1 partition (`train`: 1,165, `test`: 292) |
| `fold_2_split` | `string` | 1,457 | Official Fold 2 partition (`train`: 1,165, `test`: 292) |
| `fold_3_split` | `string` | 1,457 | Official Fold 3 partition (`train`: 1,165, `test`: 292) |
| `fold_4_split` | `string` | 1,457 | Official Fold 4 partition (`train`: 1,165, `test`: 292) |

---

## 2. Image Dimensions & Resolutions

ACNE04 contains 112 distinct image dimensions, with 76.0% sharing a high-resolution portrait format:

- **Dominant Dimension**: 3112 × 3456 pixels (1,107 images, 76.0%)
- **Secondary Dimension**: 4912 × 3264 pixels (119 images, 8.2%)
- **Aspect Ratio Distribution**:
  - $0.90$ (Height > Width portrait): 1,109 images (76.1%)
  - $1.50$ (Width > Height landscape): 142 images (9.7%)
  - $0.75$ (4:3 portrait): 71 images (4.9%)
- **Color Mode**: 100% RGB (0 grayscale, 0 CMYK, 0 RGBA).

---

## 3. Label Consistency & Hayashi Criteria

The clinical grading of ACNE04 is based on the Hayashi criteria:

$$\text{Level} = \begin{cases} 0 \text{ (Mild)} & \text{if } \text{count} \le 5 \\ 1 \text{ (Moderate)} & \text{if } 6 \le \text{count} \le 20 \\ 2 \text{ (Severe)} & \text{if } 21 \le \text{count} \le 50 \\ 3 \text{ (Very Severe)} & \text{if } \text{count} > 50 \end{cases}$$

### Findings:
1. **Level $\leftrightarrow$ Severity Mapping**: 100% consistent across all 1,457 records:
   - Level 0: 513 images (35.21%) $\rightarrow$ `mild`
   - Level 1: 633 images (43.45%) $\rightarrow$ `moderate`
   - Level 2: 182 images (12.49%) $\rightarrow$ `severe`
   - Level 3: 129 images (8.85%) $\rightarrow$ `very_severe`
2. **Hayashi Rule Violations**: **0**. Every single image's `level` matches its `lesion_count_label`.
3. **`lesion_count_label` vs. `num_lesions_annotated`**:
   - In 1,436 images (98.56%), `lesion_count_label == num_lesions_annotated`.
   - In **21 images** (1.44%), there is a small numerical discrepancy (ranging from $-10$ to $+3$).
   - All 21 discrepancies occur exclusively in **Level 3 (Very Severe)**, where both counts exceed 50. Therefore, the discrepancy has **zero impact on the severity classification**.

---

## 4. Bounding Box Integrity & Coordinate Audit

All 18,983 bounding boxes across the dataset were verified for spatial and geometric validity:

- **Total Annotated Boxes**: 18,983 (average $13.03$ lesions per image).
- **Images with Zero Lesions**: 0 (every image in the dataset has at least 1 annotated lesion).
- **Inverted Coordinates ($xmin \ge xmax$ or $ymin \ge ymax$)**: **0** (100% valid).
- **Out of Bounds Coordinates ($< 0$ or $> \text{dimension}$)**: **0** (100% valid).
- **Box Count Mismatch (`len(boxes) != num_lesions_annotated`)**: **0** (100% valid).
- **Bounding Box Area Range**: Min $16 \text{ px}^2$, Max $241,081 \text{ px}^2$.

---

## 5. Duplicate Clusters & Official Five-Fold Leakage

Computing SHA256 exact pixel hashes and 64-bit difference hashes revealed **38 exact duplicate pairs (76 images)**:

### Official Five-Fold Cross-Validation Audit:
Every official fold supplied in the dataset contains duplicate leakage across its train and test partitions:

| Fold Name | Train Count | Test Count | Duplicate Groups Crossing Train / Test | Leakage Detected? |
| :--- | :---: | :---: | :---: | :---: |
| **Original Split** | 1,165 | 292 | **12 groups** (24 images) | **Yes** |
| **Fold 0 (`fold_0_split`)** | 1,165 | 292 | **12 groups** (24 images) | **Yes** |
| **Fold 1 (`fold_1_split`)** | 1,165 | 292 | **15 groups** (30 images) | **Yes** |
| **Fold 2 (`fold_2_split`)** | 1,165 | 292 | **12 groups** (24 images) | **Yes** |
| **Fold 3 (`fold_3_split`)** | 1,165 | 292 | **11 groups** (22 images) | **Yes** |
| **Fold 4 (`fold_4_split`)** | 1,165 | 292 | **12 groups** (24 images) | **Yes** |

**Conclusion**: The official 5-fold cross-validation splits provided by the dataset authors cannot be used for unbiased academic evaluation without group-aware deduplication.

---

## 6. Verification of `clean_splits_manifest.json`

The group-stratified manifest generated for Skinora was audited directly against the master dataset index:

- **Train Split**: 990 images
- **Validation Split**: 175 images
- **Test Split**: 292 images
- **Total Samples**: 1,457 (100% accounted for, 0 omitted)
- **Hash Overlap across Splits**:
  - $\text{Train} \cap \text{Validation}$: **0**
  - $\text{Train} \cap \text{Test}$: **0**
  - $\text{Validation} \cap \text{Test}$: **0**
- **Duplicate Cluster Isolation**: **100%** (0 duplicate groups split across partitions).
- **Contradictory Label Handling**: All 8 conflicting duplicate groups are safely quarantined in the training partition, preserving test set purity.

---

## 7. Master Metadata Manifest Fields

[`backend/data/acne04_metadata/acne04_master_manifest.json`](./acne04_master_manifest.json) contains full per-image records:

```json
{
  "index": 0,
  "file_name": "levle0_0.jpg",
  "level": 0,
  "severity": "mild",
  "lesion_count_label": 3,
  "num_lesions_annotated": 3,
  "dimensions": { "width": 3112, "height": 3456, "aspect_ratio": 0.9, "mode": "RGB" },
  "sha256": "4e73ba657805d...",
  "dhash": 13917462057201660928,
  "split": "train",
  "folds": { "fold_0": "train", "fold_1": "train", "fold_2": "train", "fold_3": "test", "fold_4": "train" },
  "box_count": 3,
  "boxes": [{ "xmin": 223, "ymin": 290, "xmax": 251, "ymax": 306, "area": 448 }, ...],
  "duplicate_cluster_id": 0,
  "is_exact_duplicate": false,
  "has_label_conflict": false,
  "clean_split": "train"
}
```

