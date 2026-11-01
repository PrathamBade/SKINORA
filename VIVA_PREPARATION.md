# SKINORA — Viva & Project Defense Preparation Guide

This guide prepares you for technical and domain-specific questions from examiners during your final-year engineering defense.

---

## 1. Core Machine Learning & Computer Vision Questions

### Q1: Why did you choose ResNet50 rather than a simple CNN or a Vision Transformer?
- **Answer**:
  - Training deep networks from scratch requires hundreds of thousands of labeled images. With ACNE04 containing only 1,457 photographs, a CNN trained from scratch would severely overfit.
  - ResNet50 features **Residual Connections (skip connections)**: \( \mathcal{F}(x) + x \), which solve the vanishing/exploding gradient problem, enabling effective training of 50 layers.
  - Vision Transformers (ViTs) lack inductive bias (translation equivariance) and generally require massive datasets (like JFT-300M or ImageNet-21K) to outperform ConvNets on small image domains. ResNet50 offers an optimal balance of representational capacity, parameter efficiency (~23.5M parameters), and rapid CPU/GPU inference latency (<50ms).

### Q2: What is Transfer Learning and how does it apply here?
- **Answer**:
  - We initialize ResNet50 with weights pretrained on **ImageNet-1K** (1.28 million natural images across 1,000 object classes).
  - The earlier convolutional layers (layers 1–3) have already learned generic low-level visual primitives: Gabor-like edge detectors, color gradients, and textures.
  - We freeze layers 1–3 and fine-tune **layer 4 and the classification head** on facial acne photographs. This adapts high-level semantic representations (papules, pustules, comedones, nodules) to the four severity grades while preserving robust low-level filters.

### Q3: Why is Accuracy alone insufficient for evaluating this model?
- **Answer**:
  - The ACNE04 dataset exhibits substantial **class imbalance**:
    - `mild` + `moderate`: 78.6% of all samples
    - `severe` + `very_severe`: 21.4% of all samples
  - A trivial model predicting only `moderate` could achieve ~43% accuracy, and one predicting either mild or moderate would achieve nearly 80% accuracy while failing 100% of severe acne cases!
  - Therefore, we report **Balanced Accuracy (61.20%)** and **Macro F1 Score (58.98%)**, which compute the unweighted arithmetic mean across all four classes, ensuring minority severe cases are weighted equally.

### Q4: How do you interpret the Confusion Matrix?
- **Answer**:
  - Misclassifications are concentrated along the **off-diagonals immediately adjacent to the main diagonal** (e.g., mild confused with moderate, or severe confused with moderate).
  - In dermatological clinical grading (e.g., Pillsbury or GAGS scales), acne severity is an **ordinal continuum** rather than four disjoint categorical buckets. Lesion counts on the boundary of grades often overlap.
  - Crucially, **extreme errors are virtually absent**: only 1 mild case was classified as very severe, and 0 very severe cases were predicted as mild. Furthermore, `very_severe` acne achieved **80.8% recall**, ensuring acute cases are flagged for professional care.

### Q5: How did you prevent Data Leakage?
- **Answer**:
  - We established a strict, immutable train/validation/test protocol.
  - The 292 test images were isolated at the start and never used for hyperparameter tuning, model architecture decisions, or early stopping.
  - All data augmentations (rotations, flips, color jitter) were applied **strictly to the training partition** during runtime data loading.
  - Validation and test evaluations used purely deterministic transforms (resize 256, center crop 224, standard normalization).

---

## 2. Software Engineering & Full-Stack Architecture Questions

### Q6: How is the model served in the backend?
- **Answer**:
  - We built an asynchronous REST API using **FastAPI** (`backend/app/main.py`).
  - The model weights (`acne_resnet50.pth`) are dynamically cached in memory upon first call using safe PyTorch deserialization (`weights_only=True`).
  - The `/api/analyze` endpoint processes uploaded photos strictly in RAM using `io.BytesIO` and Pillow, converts to tensors, and runs inference under `torch.inference_mode()` (disabling gradient graphs for speed and minimal memory footprint).
  - The image bytes are **never saved to disk**, preserving user privacy.

### Q7: How does the Frontend communicate model evaluation to users?
- **Answer**:
  - A dedicated **AI Model Evaluation** dashboard tab was integrated into the React application.
  - It fetches verified evaluation metrics directly from the `/api/model-evaluation` and `/api/model-evaluation/confusion-matrix` endpoints.
  - The UI presents real verified test metrics, per-class breakdown tables, interactive confusion matrices, and transparent clinical disclaimers.

---

## 3. Responsible AI & Ethical Boundaries

### Q8: What are the ethical and clinical limitations of Skinora?
- **Answer**:
  - **Educational Research Prototype**: Skinora is explicitly not a medical device and provides educational guidance rather than medical diagnoses.
  - **Demographic Bias**: ACNE04 is an academic research dataset primarily sampled from Asian clinical populations; model performance on varied skin tones (Fitzpatrick scale IV–VI) or atypical lighting requires further multi-center validation.
  - **Referral Safeguards**: If a user reports high severity or the model predicts severe/very severe acne, Skinora actively emphasizes the "Dermatologist Search" feature and prompts consultation with a licensed clinician.

