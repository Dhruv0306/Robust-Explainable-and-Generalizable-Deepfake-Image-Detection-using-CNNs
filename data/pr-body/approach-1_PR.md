# Approach 1: CNN Baseline — End-to-end deepfake detection pipeline

## 🔬 Research Approach / Phase Reference
Select which Approach or Course Phase this PR addresses:
- [x] **Approach 1 / Phase B:** CNN Baseline Detector
- [ ] **Approach 2 / Phase B:** Robustness under Image Transformations
- [ ] **Approach 3 / Phase C:** Explainability with Grad-CAM
- [ ] **Approach 4 / Phase C:** Generalizability to Unseen Manipulations
- [ ] **Phase D / Report:** Research / White Paper & Documentation
- [ ] **Chore / Setup / Other**

---

## 📝 Description
Provide a clear summary of the changes in this PR. What research objective does it address? Why was this solution chosen?

- Full implementation of Approach 1 (CNN Baseline) for binary deepfake image detection on the FaceForensics++ C23 dataset.
- End-to-end pipeline: dataset inspection → leakage-safe splitting (relationship graph + connected components) → frame extraction (every 4th frame) → IoU-tracked face detection/cropping with MTCNN → manifest generation → training of 3 CNN backbones with 3 seeds each (9 runs) → per-frame inference → video-level aggregation with per-manipulation metrics.
- Chosen for its status as a clean, reproducible baseline that subsequent approaches (robustness, explainability, generalization) can be evaluated against without sharing a re-implemented backbone.

## 🧪 Experimental Context & Hyperparameters (If applicable)
Details on experimental runs, models, or data changes:
* **Model Architecture:** Xception (299×299), EfficientNet-B0 (224×224), ResNet50 (224×224) via `timm`, ImageNet pretrained, single-logit head
* **Dataset Name & Split:** FaceForensics++ c23, 650 videos across Original/Deepfakes/Face2Face/FaceSwap/NeuralTextures; leakage-safe subject-level split (100/18/12 train/val/test video groups derived from 65 connected components of the target-source relationship graph)
* **Hyperparameters Changed:** AdamW lr=1e-4, weight decay=1e-4, batch size 16 (≥299 input) / 32 (224 input) on GPU / 8 on CPU, max 30 epochs, early stopping patience=5, ReduceLROnPlateau patience=2, BCEWithLogitsLoss with class-weight `pos_weight`, AMP enabled when CUDA is usable
* **Transformation details/Severity levels:** N/A (clean baseline only — Approach 2 will add JPEG/blur/noise perturbations)

## 📊 Results Summary & Metrics (If applicable)
If this PR includes experimental runs, summarize the key results (provide metrics table/comparison with baseline if relevant):

| Model (Mean across seeds) | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| **Xception** | 0.9583 ± 0.0417 | 0.9745 ± 0.0219 | 0.9589 ± 0.0417 | 0.9466 ± 0.0417 | 0.9722 ± 0.0417 |
| **EfficientNet-B0** | 0.8750 ± 0.0417 | 0.9491 ± 0.0244 | 0.8682 ± 0.0493 | 0.9086 ± 0.0076 | 0.8333 ± 0.0833 |
| **ResNet50** | 0.9444 ± 0.0240 | 0.9908 ± 0.0120 | 0.9430 ± 0.0289 | 0.9744 ± 0.0442 | 0.9444 ± 0.0962 |

*Key Findings / Observations:*
- Full 9/9 experimental training matrix completed across Xception, EfficientNet-B0, and ResNet50 with 3 seeds each.
- ResNet50 achieves the most consistent performance (mean accuracy 94.4%, ROC-AUC 0.991, lowest coefficient of variation).
- Xception achieves the highest peak performance (100% accuracy and AUC at seed 2024), though with moderate seed variance.
- EfficientNet-B0 provides a lightweight alternative (87.5% mean accuracy).
- Full report available at `data/reports/approach-1-CNN-baseline-report.md`.

## 🔍 Code Changes & Quality Checks
Please verify:
- [x] Preprocessing and evaluation protocols are identical to baseline (unless validating preprocessing changes)
- [x] Random seed is set and documented for reproducibility (e.g., `seed=42`)
- [x] No data leakage (subject or video level train-test leakage has been verified)
- [x] Code follows style conventions (type annotations, structured docstrings)
- [x] Performance regressions / errors have been checked
- [ ] Visualizations / Grad-CAM overlays conform to dataviz guidelines (if applicable)

## 📎 Checklist
- [x] My code matches the existing style and conventions of the repository.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have updated the documentation or notebooks to reflect these changes.
- [ ] I have added/updated unit tests where necessary.
- [ ] All tests pass locally.
- [x] Commit messages follow the project convention (`type(scope): subject`).
