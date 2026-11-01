# SKINORA — Model Evaluation & Empirical Performance Report

This document presents the complete empirical benchmark of the Skinora ResNet50 acne-severity model evaluated against the academic **ACNE04** dataset.

---

## 1. Evaluation Methodology

### 1.1 Strict Split Separation
- **Training partition**: 990 facial photographs
- **Validation partition**: 175 facial photographs (stratified by severity)
- **Held-out Test partition**: 292 facial photographs (completely isolated)

> [!IMPORTANT]
> In accordance with scientific ML methodology, **zero test samples** were used for early stopping, hyperparameter search, or checkpoint selection. Model selection was performed strictly on **validation macro F1**.

### 1.2 Evaluation Metrics
Given the substantial class imbalance (mild and moderate comprise 78.6% of the dataset), raw accuracy alone is insufficient. We evaluate:
1. **Accuracy**: Overall fraction of correct predictions.
2. **Balanced Accuracy**: Macro-average of recall across all four classes, giving equal weight to minority classes.
3. **Macro F1 Score**: Harmonic mean of macro precision and macro recall.
4. **Weighted F1 Score**: F1 score weighted by class support.
5. **Cross-Entropy Test Loss**: Multi-class logarithmic loss.
6. **Expected Calibration Error (ECE)**: Difference between model confidence and actual accuracy across 10 confidence bins.

---

## 2. Empirical Benchmark Results

### 2.1 Overall Summary

| Metric | Score (Baseline ResNet50) |
|---|---:|
| **Model Architecture** | ResNet50 (ImageNet-1K Pretrained) |
| **Primary Dataset** | ACNE04 |
| **Best Validation Accuracy** | **72.57%** |
| **Best Validation Macro F1** | **72.57%** |
| **Test Accuracy** | **61.30%** |
| **Test Balanced Accuracy** | **61.20%** |
| **Test Macro F1 Score** | **58.98%** |
| **Test Weighted F1 Score** | **61.65%** |
| **Cross-Entropy Test Loss** | **0.8603** |
| **Expected Calibration Error (ECE)** | **0.0704** |
| **Mean Softmax Confidence** | **0.6719** |

---

## 3. Per-Class Classification Report

Evaluated on the 292 held-out test set images:

| Severity Level | Precision | Recall | F1-Score | Support (Test Images) |
|---|---:|---:|---:|---:|
| **mild** | **70.33%** | 62.14% | 0.6598 | 103 |
| **moderate** | 64.52% | 62.99% | 0.6375 | 127 |
| **severe** | 31.82% | 38.89% | 0.3500 | 36 |
| **very_severe** | 63.64% | **80.77%** | **0.7119** | 26 |
| **Macro Average** | **57.58%** | **61.20%** | **58.98%** | 292 |
| **Weighted Average** | **62.53%** | **61.30%** | **61.65%** | 292 |

---

## 4. Confusion Matrix Analysis

### Numerical Matrix (292 Test Images)
```text
               Predicted
           mild  moderate  severe  very_severe
True
mild         64        30       8            1   (103 total)
moderate     26        80      17            4   (127 total)
severe        1        14      14            7   (36 total)
very_severe   0         0       5           21   (26 total)
```

### Key Clinical & Computer Vision Observations:
1. **Adjacent-Class Confusion**:
   - 30 mild cases were classified as moderate; 26 moderate cases were classified as mild.
   - 14 severe cases were classified as moderate; 7 severe cases were classified as very severe.
   - In dermatological clinical practice, acne severity boundaries are continuous rather than discrete. Misclassifications are overwhelmingly between adjacent severity grades.
2. **Extreme Class Separation**:
   - `mild` vs `very_severe`: Only 1 mild image was predicted as very severe, and 0 very severe images were predicted as mild. The model distinguishes the clinical extremes with high reliability.
3. **High Sensitivity on Severe Cases**:
   - `very_severe` achieved an **80.77% recall** (21 of 26 cases accurately flagged), ensuring acute cases are rarely overlooked.
4. **Severe Class Challenge**:
   - `severe` (36 test images) has the lowest F1 (0.3500) because it acts as an intermediate transition band between moderate and very severe inflammatory lesions.

---

## 5. Visual Artifacts
- **Confusion Matrix**: Saved at `backend/experiments/baseline_resnet50/confusion_matrix.png`
- **Normalized Matrix**: Saved at `backend/experiments/baseline_resnet50/confusion_matrix_normalized.png`
- **API Endpoint**: Accessible in real-time via `http://127.0.0.1:8000/api/model-evaluation/confusion-matrix`

