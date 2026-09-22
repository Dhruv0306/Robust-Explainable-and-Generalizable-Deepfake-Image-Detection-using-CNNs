# Approach 3: Cross-Architecture Explainability Report

## 1. Executive Summary

This report compares explanation localization, intervention faithfulness, and transformation stability for the three fixed CNN deepfake detectors evaluated in Approach 3: Xception (seed 2024), EfficientNet-B0 (seed 2024), and ResNet50 (seed 123). Each model was evaluated on the FaceForensics++ C23 test manifest under one clean reference condition and 12 transformations shared with Approach 2.

The corrected evaluation contains **81,952 frame-condition records per architecture** (6,304 manifest rows × 13 conditions), yielding **245,856 frame-condition records** in total. Frame observations are retained for auditability and visualization. Video-category observations are the primary descriptive unit; the four category views associated with a source pair are not treated as independent source pairs in interpretation.

Three high-level patterns are consistent across architectures:

1. **Clean Grad-CAM localization is moderate rather than near-perfect.** Xception has the highest clean Saliency Overlap (SO = 0.2948), followed by ResNet50 (0.2584) and EfficientNet-B0 (0.1585).
2. **JPEG compression damages explanations far more than resizing.** At JPEG Q20, continuous explanation similarity drops to 0.1514 for Xception, 0.3273 for EfficientNet-B0, and 0.1664 for ResNet50. At resize 0.25, the corresponding similarities remain 0.8415, 0.6744, and 0.8931.
3. **Architecture-specific photometric behavior remains visible.** Xception preserves high explanation similarity under darkening 0.40 (0.9467), whereas ResNet50 declines to 0.6003 and EfficientNet-B0 to 0.6411.

These results describe transformation-conditioned behavior of Grad-CAM explanations. They do not establish that Grad-CAM maps reveal a model's complete causal reasoning process.

---

## 2. Experimental Protocol

### 2.1 Models and selected checkpoints

| Architecture | Selected seed | Seed-selection basis | Grad-CAM target layer |
| :--- | :---: | :--- | :--- |
| Xception | 2024 | Highest video-level F1; ROC-AUC tie-breaker | `conv4.pointwise` |
| EfficientNet-B0 | 2024 | Highest video-level F1; ROC-AUC tie-breaker | `conv_head` |
| ResNet50 | 123 | Highest video-level F1; ROC-AUC tie-breaker | `layer4[-1]` |

The fake-class logit was the fixed Grad-CAM target, including when a fake frame was predicted as real.

### 2.2 Test population and transformations

The test manifest contains 6,304 rows: 1,348 Original frames, 1,348 Deepfakes frames, 1,348 Face2Face frames, 1,130 FaceSwap frames, and 1,130 NeuralTextures frames. Manipulation-mask localization metrics apply only to the 4,956 fake-category rows.

Each architecture was evaluated under 13 conditions:

- Clean reference;
- JPEG compression: Q80, Q50, Q20;
- Resize: 0.75, 0.50, 0.25;
- Darkening: 0.80, 0.60, 0.40;
- Brightening: 1.20, 1.40, 1.60.

### 2.3 Metrics and analysis boundaries

Localization metrics use the top 20% Grad-CAM activation region as the primary threshold, with 10% and 30% retained as sensitivity outputs:

\[
SO = \frac{|S \cap M|}{|S|}, \qquad
IoU = \frac{|S \cap M|}{|S \cup M|}
\]

Additional metrics are continuous saliency mass inside the manipulation mask, maximum-activation hit rate, intervention faithfulness, cosine explanation similarity, and binary explanation IoU.

Faithfulness reports the within-input intervention:

\[
\Delta P_{fake} = P_{fake}(x) - P_{fake}(x_{masked})
\]

For high-resolution face crops, Grad-CAM may be computed at a capped resolution for memory safety, but its salient mask is mapped back to the original transformed crop before masking. Thus the baseline and masked probability use the same model preprocessing pathway.

---

## 3. Clean Baseline Comparison

Table 1 summarizes clean explanation behavior on manipulated video-category observations.

**Table 1. Clean explanation localization and intervention faithfulness**

| Architecture | SO@20 | IoU@20 | Saliency mass | Hit rate | ΔP blur | ΔP zero | ΔP mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Xception | 0.2948 ± 0.1704 | 0.1919 ± 0.1074 | 0.2918 ± 0.1659 | 31.88% | 0.0032 | 0.6080 | 0.4206 |
| EfficientNet-B0 | 0.1585 ± 0.1114 | 0.0915 ± 0.0660 | 0.1450 ± 0.0995 | 18.80% | 0.1807 | 0.2970 | 0.2923 |
| ResNet50 | 0.2584 ± 0.1354 | 0.1557 ± 0.0785 | 0.2582 ± 0.1317 | 33.22% | 0.0452 | 0.6445 | 0.5169 |

Xception provides the strongest clean spatial correspondence with known manipulated regions, while ResNet50 has the highest maximum-activation hit rate. EfficientNet-B0 shows weaker mask localization but a larger blur-based probability change, demonstrating that intervention behavior and spatial alignment are distinct explanation properties.
