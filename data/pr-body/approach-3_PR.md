## 🔬 Research Approach / Phase Reference

Select which Approach or Course Phase this PR addresses:

* [ ] **Approach 1 / Phase B:** CNN Baseline Detector
* [ ] **Approach 2 / Phase B:** Robustness under Image Transformations
* [x] **Approach 3 / Phase C:** Explainability with Grad-CAM
* [ ] **Approach 4 / Phase C:** Generalizability to Unseen Manipulations
* [ ] **Phase D / Report:** Research / White Paper & Documentation
* [ ] **Chore / Setup / Other**

---

## 📝 Description

This PR implements Approach 3, which evaluates the explainability of the CNN-based deepfake detectors developed in Approach 1 and evaluates how their explanations change under the controlled transformations introduced in Approach 2.

The implementation uses Grad-CAM to evaluate three explanation properties:

* **Localization:** agreement between Grad-CAM saliency and the FaceForensics++ manipulation mask using Saliency Overlap (SO), Regional IoU, saliency mass, and mask hit rate.
* **Faithfulness:** change in fake-class probability after masking the regions identified by Grad-CAM.
* **Explanation stability:** similarity between clean and transformed Grad-CAM maps using cosine similarity and explanation IoU.

The analysis is performed for Xception, EfficientNet-B0, and ResNet50 using the selected seeds from Approach 1. The same test population and preprocessing protocol are retained, and no model is retrained or fine-tuned for explainability.

The implementation also adds prediction-state analysis and integrates Approach 2 detector degradation with Approach 3 explanation degradation to examine whether detector performance changes are associated with changes in explanation properties.

---

## 🧪 Experimental Context & Hyperparameters (If applicable)

* **Model Architecture:** Xception, EfficientNet-B0, ResNet50
* **Dataset Name & Split:** FaceForensics++ C23, same test split used in Approaches 1 and 2
* **Selected Seeds:** Xception = 2024, EfficientNet-B0 = 2024, ResNet50 = 123
* **Input Resolution:** Xception = 299×299, EfficientNet-B0 = 224×224, ResNet50 = 224×224
* **Grad-CAM Target Layers:** Xception = `conv4.pointwise`, EfficientNet-B0 = `conv_head`, ResNet50 = `layer4[-1]`
* **Grad-CAM Target:** Fake-class logit for all samples, independent of predicted class
* **Primary Localization Threshold:** Top 20% Grad-CAM activation
* **Localization Sensitivity Checks:** Top 10% and 30% activation thresholds
* **Faithfulness Masking:** Blur, zero, and mean masking, with blur used as the primary masking condition
* **Explanation Stability:** Clean-to-transformed Grad-CAM cosine similarity and explanation IoU
* **Transformations:** JPEG Q80, Q50, Q20; resizing 0.75, 0.50, 0.25; darkening 0.80, 0.60, 0.40; brightening 1.20, 1.40, 1.60
* **Test Frames:** 6,304 frames per model
* **Frame-condition Evaluations:** 81,952 per model, 245,856 across all three models
* **Source IDs:** 24 total, consisting of 12 original source IDs and 12 manipulated source-pair IDs
* **Statistical Analysis:** Paired Wilcoxon signed-rank tests, rank-biserial effect sizes, bootstrap 95% confidence intervals, Spearman correlation, and Benjamini-Hochberg FDR correction

## 📊 Results Summary & Metrics (If applicable)

The clean-condition localization results were:

| Model               | Saliency Overlap | Regional IoU | Saliency Mass | Mask Hit Rate |
| ------------------- | ---------------: | -----------: | ------------: | ------------: |
| **Xception**        |           0.2948 |       0.1919 |        0.2918 |        31.88% |
| **ResNet50**        |           0.2584 |       0.1557 |        0.2582 |        33.22% |
| **EfficientNet-B0** |           0.1585 |       0.0915 |        0.1450 |        18.80% |

Clean-condition faithfulness using the three masking methods produced:

| Model               |   Blur |   Zero |   Mean |
| ------------------- | -----: | -----: | -----: |
| **Xception**        | 0.0032 | 0.6080 | 0.4206 |
| **EfficientNet-B0** | 0.1807 | 0.2970 | 0.2923 |
| **ResNet50**        | 0.0452 | 0.6445 | 0.5169 |

Explanation stability decreased as transformation severity increased for several transformation families, with JPEG compression producing particularly large changes in Grad-CAM similarity.

For example, Xception's clean SO and explanation cosine similarity were 0.2948 and 1.0000. Under JPEG compression, these values decreased to:

| Condition |     SO | Explanation cosine similarity |
| --------- | -----: | ----------------------------: |
| Clean     | 0.2948 |                        1.0000 |
| JPEG Q80  | 0.2662 |                        0.8624 |
| JPEG Q50  | 0.1751 |                        0.5363 |
| JPEG Q20  | 0.0883 |                        0.1519 |

The prediction-state analysis also showed that prediction preservation does not by itself guarantee strong localization. For Xception:

| Prediction state      |     SO | Explanation cosine similarity |
| --------------------- | -----: | ----------------------------: |
| Correct → Correct     | 0.3169 |                        0.9581 |
| Correct → Incorrect   | 0.0781 |                        0.2248 |
| Incorrect → Correct   | 0.3089 |                        0.6167 |
| Incorrect → Incorrect | 0.1140 |                        0.9088 |

The Approach 2 and Approach 3 integration showed positive associations between detector F1 degradation and explanation stability degradation across the evaluated transformations. The strongest relationship was observed for EfficientNet-B0, with Spearman ρ = 0.958 for detector F1 degradation versus explanation stability degradation, with BH-adjusted p < 0.001.

For confidence shifts, all three architectures showed strong positive relationships between fake-probability degradation and localization degradation. For Xception, the relationship between fake-probability degradation and SO degradation was ρ = 0.972, with BH-adjusted p < 0.001.

*Key Findings / Observations:*

* Grad-CAM localization did not uniformly align with the ground-truth manipulation regions across architectures.
* Xception obtained the highest clean Saliency Overlap among the three models, while ResNet50 had the highest mask hit rate.
* JPEG compression produced substantial decreases in both localization and explanation stability, particularly at Q50 and Q20.
* Resizing produced smaller explanation changes than severe JPEG compression for all three models, although the magnitude varied by architecture.
* Brightening and darkening produced architecture-dependent changes in localization and explanation stability.
* Prediction preservation and explanation quality were not equivalent. Some prediction-preserved cases retained high explanation similarity while showing relatively low localization agreement.
* Detector-performance degradation and explanation stability degradation were positively associated across the evaluated transformations.
* The confidence-shift analysis showed stronger and more consistent relationships with localization degradation than the detector F1 analysis.
* Global Grad-CAM heatmaps were generated for the overall manipulated population and for each manipulation category as descriptive visual analyses.

## 🔍 Code Changes & Quality Checks

Please verify:

* [x] Preprocessing and evaluation protocols are identical to baseline (unless validating preprocessing changes)
* [x] Random seed is set and documented for reproducibility (e.g., `seed=42`)
* [x] No data leakage (subject or video level train-test leakage has been verified)
* [x] Code follows style conventions (type annotations, structured docstrings)
* [x] Performance regressions / errors have been checked
* [x] Visualizations / Grad-CAM overlays conform to dataviz guidelines (if applicable)

Additional validation performed:

* [x] 81,952 frame-condition rows validated per model
* [x] 780 video-level rows validated per model
* [x] 24 source IDs validated per model
* [x] 576 valid transformed explanation cosine-similarity rows validated per model
* [x] 576 valid transformed explanation-IoU rows validated per model
* [x] No transformed fake samples with missing Saliency Overlap
* [x] Manipulation-mask alignment checked using the same expanded face bounding box used during preprocessing
* [x] Category-aware Grad-CAM caching verified
* [x] Clean/transformed explanation map dimensions validated
* [x] Explicit Grad-CAM and CUDA memory cleanup implemented
* [x] Post-hoc analysis scripts syntax checked
* [x] BH-FDR correction applied to predefined statistical families

## 📎 Checklist

* [x] My code matches the existing style and conventions of the repository.
* [x] I have commented my code, particularly in hard-to-understand areas.
* [x] I have updated the documentation or notebooks to reflect these changes.
* [ ] I have added/updated unit tests where necessary.
* [x] All tests pass locally.
* [x] Commit messages follow the project convention (`type(scope): subject`).

<!--
🤖 Generated with Claude Code
-->
