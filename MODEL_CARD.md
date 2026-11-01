# Model Card: Skinora ResNet50 Acne Severity Classifier

## 1. Model Details
- **Model Name**: Skinora ResNet50 Acne Classifier
- **Architecture**: Deep Residual Convolutional Neural Network (ResNet50, He et al., 2016)
- **Base Pretraining**: ImageNet-1K (via PyTorch torchvision default weights)
- **Fine-Tuning Dataset**: ACNE04 (1,457 clinical face photos)
- **License / Terms**: Academic and engineering research prototype. Non-commercial use.
- **Framework**: PyTorch 2.x, Torchvision, Scikit-Learn, FastAPI

---

## 2. Intended Use
- **Primary Use**: Educational skincare exploration and self-check monitoring prototype for estimating facial acne severity across four categories: `mild`, `moderate`, `severe`, and `very_severe`.
- **Target Audience**: Students, researchers, and end-users seeking personalized educational skincare routine guidance.

---

## 3. Prohibited & Out-of-Scope Uses
- **Not a Medical Device**: This model is **NOT** a diagnostic tool, medical device, or replacement for professional clinical consultation.
- **No Pathological Diagnosis**: Does not diagnose skin cancers, melanoma, cysts, rosacea, eczema, dermatitis, or fungal infections.
- **No Treatment Prescriptions**: Model predictions must never be used to initiate, alter, or discontinue prescription medications.

---

## 4. Training Data & Preprocessing
- **Source**: ACNE04 (Wu et al., ICCV 2019; hosted on Hugging Face by Layered-Labs).
- **Resolution**: 224 × 224 pixels center crop (resized from 256).
- **Normalization**: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Augmentation**: Random horizontal flips, rotation (±12°), color jitter (brightness, contrast), random resized crops (scale 0.85–1.0).

---

## 5. Quantitative Performance
- **Validation Accuracy**: 72.57%
- **Validation Macro F1**: 72.57%
- **Test Accuracy**: 61.30%
- **Balanced Accuracy**: 61.20%
- **Macro F1 Score**: 58.98%
- **Expected Calibration Error (ECE)**: 0.0704

---

## 6. Known Limitations & Bias Considerations
1. **Dataset Demographics**: ACNE04 predominantly comprises Chinese clinical facial photographs. Generalization across Fitzpatrick skin types IV–VI, different ethnicities, everyday camera sensors, and varied smartphone lighting conditions may vary.
2. **Class Imbalance**: Classes 0 and 1 represent 78.6% of the dataset; minority class `severe` exhibits lower precision (31.8%) due to intermediate clinical ambiguity.
3. **In-Memory Processing**: Photos are processed exclusively in RAM during inference and immediately discarded. No facial photos are persisted to disk or database.

