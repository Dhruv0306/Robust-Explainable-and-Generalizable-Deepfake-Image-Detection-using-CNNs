# Approach 3: Explainability, faithfulness, and explanation robustness

## 1. Overview

Approach 3 evaluates whether the CNN detectors provide explanations that are spatially aligned with manipulated regions, remain stable under controlled input transformations, and reflect the model's prediction behavior.

The analysis uses Grad-CAM to generate local explanations for the fake-class logit. The evaluation is performed for Xception, EfficientNet-B0, and ResNet50 using the same FaceForensics++ C23 test split used in Approaches 1 and 2.

The analysis addresses four main questions:

1. How well do Grad-CAM explanations localize the manipulated regions?
2. How faithful are the highlighted regions to the model's fake prediction?
3. How stable are explanations when the input is transformed?
4. How do explanation changes relate to detector-performance degradation and prediction changes?

The analysis also connects Approach 3 with the robustness results from Approach 2, allowing detector performance and explanation behavior to be studied under the same transformation conditions.

---

## 2. Dataset and experimental population

The experiments use the FaceForensics++ C23 dataset with the following categories:

* Original
* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

The same test split from Approach 1 is retained. Frames are sampled every fourth frame, corresponding to an effective sampling rate of 7.5 fps.

The final test population contains 6,304 frames per model:

| Category       | Test frames |
| -------------- | ----------: |
| Original       |       1,348 |
| Deepfakes      |       1,348 |
| Face2Face      |       1,348 |
| FaceSwap       |       1,130 |
| NeuralTextures |       1,130 |
| **Total**      |   **6,304** |

There are 4,956 manipulated-frame observations per model.

At the video level, the experiment contains 24 source-video identifiers:

* 12 Original source videos
* 12 manipulated source-pair identifiers

The manipulated source pairs are evaluated across the four manipulation categories, resulting in 60 category-video units.

Each model is evaluated under 13 conditions:

* 1 clean condition
* 12 transformed conditions

This produces 81,952 frame-condition evaluations per model and 245,856 evaluations across all three models.

For paired quantitative explainability analysis, the four manipulation-category observations originating from the same source pair are aggregated to the 12 manipulated source-pair level. Prediction-state analysis remains at the category-video level.

---

## 3. Models and Grad-CAM configuration

The three CNN architectures from Approach 1 are evaluated using the selected seeds:

| Model           | Seed | Input size | Grad-CAM target   |
| --------------- | ---: | ---------: | ----------------- |
| Xception        | 2024 |  299 × 299 | `conv4.pointwise` |
| EfficientNet-B0 | 2024 |  224 × 224 | `conv_head`       |
| ResNet50        |  123 |  224 × 224 | `layer4[-1]`      |

The model produces one logit for binary classification. The fake probability is obtained using the sigmoid function.

Grad-CAM targets the fake-class logit for every image, regardless of whether the model's final prediction is real or fake. This provides a consistent explanation target across correct and incorrect predictions.

The implementation resolves the final suitable convolutional layer according to the architecture rather than assuming the same layer structure for all models.

---

## 4. Face preprocessing and manipulation masks

The explainability analysis uses the same face preprocessing procedure as the detector evaluation.

The selected face is extracted using the existing expanded face bounding box. The same bounding box is used to transform the FaceForensics++ manipulation mask into the corresponding face-crop coordinates.

Manipulation masks are available for:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

Original images do not have manipulation masks and are therefore excluded from quantitative manipulation-localization analysis.

The currently enabled transformations preserve the final face-crop dimensions. JPEG compression and brightness changes preserve the image geometry directly. Resizing first downsamples the image and then restores it to the original dimensions. Therefore, the same transformed face-crop mask remains spatially aligned for the current transformation set.

---

## 5. Transformation conditions

Approach 3 uses the same fixed transformation conditions as Approach 2:

| Transformation family | Conditions       |
| --------------------- | ---------------- |
| JPEG compression      | Q80, Q50, Q20    |
| Resizing              | 0.75, 0.50, 0.25 |
| Darkening             | 0.80, 0.60, 0.40 |
| Brightening           | 1.20, 1.40, 1.60 |

The following transformations remain disabled:

* Noise
* Blur
* Contrast
* Crop

This provides a controlled setting in which detector performance and explanation behavior can be compared using identical input perturbations.

---

## 6. Local explanation metrics

### 6.1 Saliency Overlap

Grad-CAM maps are normalized to the range [0, 1]. A binary saliency mask is generated using the top 20% of Grad-CAM activations.

Saliency Overlap (SO) measures the proportion of the predicted salient region that falls inside the ground-truth manipulation mask:

$$
SO = \frac{|S \cap M|}{|S|}
$$

where:

* \(S\) is the Grad-CAM saliency mask
* \(M\) is the manipulation mask

Higher SO indicates greater spatial agreement between the highlighted region and the manipulation mask.

Sensitivity analyses are also performed using the top 10% and top 30% activation thresholds.

### 6.2 Regional IoU

Regional Intersection over Union measures the overlap between the Grad-CAM saliency mask and the manipulation mask:

$$
IoU = \frac{|S \cap M|}{|S \cup M|}
$$

SO and IoU capture related but different properties. SO measures how much of the selected salient region is inside the manipulation mask, while IoU also penalizes differences between the overall spatial extents of the two masks.

### 6.3 Additional localization measures

The analysis also records:

* Saliency mass inside the manipulation mask
* Maximum-activation mask hit rate

These provide complementary measurements of whether the strongest Grad-CAM activations occur within manipulated regions.

---

## 7. Clean explanation results

The clean test condition provides the baseline for the transformation analysis.

| Model           |     SO |    IoU | Saliency mass | Hit rate |
| --------------- | -----: | -----: | ------------: | -------: |
| Xception        | 0.2948 | 0.1919 |        0.2918 |   31.88% |
| EfficientNet-B0 | 0.1585 | 0.0915 |        0.1450 |   18.80% |
| ResNet50        | 0.2584 | 0.1557 |        0.2582 |   33.22% |

The results show that Grad-CAM localization does not consistently align with the manipulation masks, including for correctly classified manipulated images.

For Xception, the category-level SO values provide additional variation:

| Manipulation   |     SO |          Hit rate |
| -------------- | -----: | ----------------: |
| Deepfakes      | 0.3899 |            41.99% |
| Face2Face      | 0.3415 | Not reported here |
| NeuralTextures | 0.2829 | Not reported here |
| FaceSwap       | 0.1730 |            15.66% |

These values indicate that localization quality varies across manipulation categories.

---

## 8. Faithfulness analysis

Faithfulness evaluates whether the regions highlighted by Grad-CAM are related to the model's fake prediction.

The primary measure is the change in fake probability after masking the Grad-CAM-selected region:

$$
\Delta P_{fake}
=
P_{fake}(x)
-
P_{fake}(x_{masked})
$$

A larger positive value means that masking the selected region produces a larger reduction in fake probability.

Three masking methods are evaluated:

* Blur
* Zero
* Mean-value replacement

Blur is used as the primary masking condition, while zero and mean masking provide sensitivity analysis.

Clean-condition faithfulness values are:

| Model           |   Blur |   Zero |   Mean |
| --------------- | -----: | -----: | -----: |
| Xception        | 0.0032 | 0.6080 | 0.4206 |
| EfficientNet-B0 | 0.1807 | 0.2970 | 0.2923 |
| ResNet50        | 0.0452 | 0.6445 | 0.5169 |

The masking method therefore affects the magnitude of the measured confidence change. For this reason, the masking method is retained explicitly rather than treating the three values as interchangeable.

---

## 9. Explanation stability under transformations

Explanation stability measures whether the spatial explanation remains similar after an input transformation.

For each clean/transformed pair, the normalized Grad-CAM maps are compared using cosine similarity:

$$
ES_{cos}
=
\frac{G_c \cdot G_t}
{\|G_c\|\|G_t\|}
$$

where \(G_c\) is the clean Grad-CAM map and \(G_t\) is the transformed Grad-CAM map.

Explanation IoU is also calculated from binary explanation masks.

These are direct clean-to-transformed metrics. A clean Grad-CAM map does not have an `ES_cos` value against itself, so a clean-versus-transformed statistical test is not applied to `ES_cos`.

The main transformation results are:

| Transformation | Xception SO | Xception ES | EfficientNet SO | EfficientNet ES | ResNet SO | ResNet ES |
| -------------- | ----------: | ----------: | --------------: | --------------: | --------: | --------: |
| Clean          |      0.2948 |      1.0000 |          0.1585 |          1.0000 |    0.2584 |    1.0000 |
| JPEG Q80       |      0.2662 |      0.8624 |          0.1241 |          0.7220 |    0.2420 |    0.9219 |
| JPEG Q50       |      0.1751 |      0.5363 |          0.0754 |          0.4605 |    0.1591 |    0.5736 |
| JPEG Q20       |      0.0883 |      0.1519 |          0.0571 |          0.3273 |    0.0627 |    0.1664 |
| Resize 0.75    |      0.3135 |      0.9543 |          0.1758 |          0.8950 |    0.2686 |    0.9712 |
| Resize 0.50    |      0.3033 |      0.9490 |          0.1725 |          0.8413 |    0.2703 |    0.9637 |
| Resize 0.25    |      0.2712 |      0.8418 |          0.1738 |          0.6744 |    0.2562 |    0.8931 |
| Dark 0.80      |      0.2974 |      0.9934 |          0.1857 |          0.9227 |    0.2466 |    0.9329 |
| Dark 0.60      |      0.3043 |      0.9762 |          0.1981 |          0.8148 |    0.2084 |    0.7972 |
| Dark 0.40      |      0.3160 |      0.9477 |          0.1823 |          0.6411 |    0.1671 |    0.6003 |
| Bright 1.20    |      0.2849 |      0.9714 |          0.1320 |          0.8513 |    0.2536 |    0.9735 |
| Bright 1.40    |      0.2619 |      0.9201 |          0.0971 |          0.7264 |    0.2324 |    0.8999 |
| Bright 1.60    |      0.2062 |      0.7810 |          0.0871 |          0.6211 |    0.1869 |    0.7818 |

JPEG compression produces the largest decrease in explanation similarity across the tested conditions. At Q20, the mean cosine similarity is 0.1519 for Xception, 0.3273 for EfficientNet-B0, and 0.1664 for ResNet50.

Resizing produces smaller explanation changes for Xception and ResNet50 than the JPEG conditions. EfficientNet-B0 shows a larger reduction in explanation similarity as the resolution decreases.

Brightness transformations produce architecture-dependent changes. Xception retains relatively high similarity under darkening, while ResNet50 and EfficientNet-B0 show larger decreases at stronger darkening levels.

---

## 10. Prediction-state analysis

Explanation behavior is also conditioned on whether the transformation changes the model prediction.

Each clean/transformed observation is classified into one of four states:

1. Correct → Correct
2. Correct → Incorrect
3. Incorrect → Correct
4. Incorrect → Incorrect

For Xception, the observed explanation statistics are:

| Prediction state      |     SO |     ES |
| --------------------- | -----: | -----: |
| Correct → Correct     | 0.3169 | 0.9581 |
| Correct → Incorrect   | 0.0781 | 0.2248 |
| Incorrect → Correct   | 0.3089 | 0.6167 |
| Incorrect → Incorrect | 0.1140 | 0.9088 |

The Correct → Correct state has relatively high localization and explanation similarity.

The Correct → Incorrect state has substantially lower localization and explanation similarity, indicating that transformations associated with a prediction change can also produce large changes in the spatial explanation.

The Incorrect → Incorrect state provides a different pattern. Explanation similarity remains high while localization agreement with the manipulation mask remains low. Thus, explanation stability alone does not establish that the explanation is spatially correct.

This analysis separates two properties that can otherwise be conflated:

* stability of an explanation
* agreement of an explanation with the manipulation mask

---

## 11. Transformation-conditioned explanation behavior

### JPEG compression

JPEG compression produces the clearest degradation in both localization and explanation similarity.

For Xception, SO decreases from 0.2948 in the clean condition to 0.0883 at Q20, while ES decreases from 1.0000 to 0.1519.

ResNet50 follows a similar pattern, with SO decreasing from 0.2584 to 0.0627 and ES decreasing from 1.0000 to 0.1664.

EfficientNet-B0 also shows decreasing localization and stability, although its ES at Q20 remains higher than the corresponding Xception and ResNet50 values.

### Resizing

The resizing conditions generally produce smaller changes than JPEG compression.

Xception maintains ES values of 0.9543, 0.9490, and 0.8418 at 0.75, 0.50, and 0.25 scale factors.

ResNet50 shows a similar pattern, with ES values of 0.9712, 0.9637, and 0.8931.

EfficientNet-B0 shows a larger decrease, reaching 0.6744 at the 0.25 condition.

### Darkening

Darkening produces architecture-specific behavior.

Xception SO increases slightly from 0.2948 to 0.3160 across the tested darkening conditions.

ResNet50 SO decreases from 0.2584 to 0.1671 at the strongest darkening condition.

EfficientNet-B0 exhibits a non-monotonic SO pattern.

The results therefore do not support treating brightness changes as having a uniform effect across architectures.

### Brightening

Brightening generally reduces both localization and explanation similarity as the transformation becomes stronger.

For Xception, SO decreases from 0.2849 at 1.20 to 0.2062 at 1.60.

EfficientNet-B0 decreases from 0.1320 to 0.0871.

ResNet50 decreases from 0.2536 to 0.1869.

---

## 12. Localization and faithfulness relationship

The analysis examines whether better localization is associated with stronger faithfulness.

For each transformation, the analysis compares localization metrics against the change in fake probability after masking.

For the Xception clean source-level analysis, the observed Spearman relationships were:

| Localization measure               | Spearman ρ | p-value |
| ---------------------------------- | ---------: | ------: |
| SO vs blur faithfulness            |     -0.070 |   0.829 |
| IoU vs blur faithfulness           |     -0.119 |   0.713 |
| Saliency mass vs blur faithfulness |     -0.042 |   0.897 |

The clean Xception results therefore do not show a strong monotonic relationship between localization agreement and blur-based fake-probability change.

This indicates that spatial agreement with the manipulation mask and faithfulness to the model's prediction measure different properties of the explanation.

---

## 13. Approach 2 and Approach 3 integration

Approach 2 provides detector-performance degradation under controlled transformations. Approach 3 provides explanation degradation under the same transformations.

The following quantities are used:

$$
D_{F1}
=
F1_{clean}
-
F1_{transformed}
$$

$$
D_{SO}
=
SO_{clean}
-
SO_{transformed}
$$

$$
D_{IoU}
=
IoU_{clean}
-
IoU_{transformed}
$$

$$
D_{stability}
=
1
-
ES_{cos}
$$

For faithfulness:

$$
D_{faith}
=
Faith_{clean}
-
Faith_{transformed}
$$

Fake-probability shift is kept as a separate quantity:

$$
D_{prob}
=
P_{fake,clean}
-
P_{fake,transformed}
$$

\(D_{prob}\) represents a confidence shift and is not treated as equivalent to detector-performance degradation.

### Detector-performance relationships

The Spearman relationships between Approach 2 F1 degradation and Approach 3 explanation degradation are:

| Model           | Relationship        |      ρ | BH-adjusted p |
| --------------- | ------------------- | -----: | ------------: |
| Xception        | D_F1 vs D_SO        |  0.601 |        0.0516 |
| Xception        | D_F1 vs D_IoU       |  0.601 |        0.0516 |
| Xception        | D_F1 vs D_stability |  0.861 |        0.0019 |
| Xception        | D_F1 vs D_faith     | -0.109 |        0.7360 |
| EfficientNet-B0 | D_F1 vs D_SO        |  0.718 |        0.0204 |
| EfficientNet-B0 | D_F1 vs D_IoU       |  0.718 |        0.0204 |
| EfficientNet-B0 | D_F1 vs D_stability |  0.958 |      0.000012 |
| EfficientNet-B0 | D_F1 vs D_faith     |  0.451 |        0.1543 |
| ResNet50        | D_F1 vs D_SO        |  0.601 |        0.0516 |
| ResNet50        | D_F1 vs D_IoU       |  0.601 |        0.0516 |
| ResNet50        | D_F1 vs D_stability |  0.847 |        0.0020 |
| ResNet50        | D_F1 vs D_faith     |  0.576 |        0.0597 |

The strongest and most consistent relationship is between detector-performance degradation and explanation stability degradation. The relationship is statistically significant after BH-FDR correction for all three architectures.

The SO and IoU relationships are significant for EfficientNet-B0, while the corresponding Xception and ResNet50 relationships are close to but above the 0.05 BH-adjusted threshold.

The faithfulness relationships do not reach the 0.05 BH-adjusted threshold for any architecture.

### Confidence-shift relationships

The corresponding relationships using fake-probability shift are:

| Model           | Relationship          |     ρ | BH-adjusted p |
| --------------- | --------------------- | ----: | ------------: |
| Xception        | D_prob vs D_SO        | 0.972 |        <0.001 |
| Xception        | D_prob vs D_IoU       | 0.972 |        <0.001 |
| Xception        | D_prob vs D_stability | 0.776 |        0.0040 |
| Xception        | D_prob vs D_faith     | 0.252 |        0.4690 |
| EfficientNet-B0 | D_prob vs D_SO        | 0.811 |        0.0020 |
| EfficientNet-B0 | D_prob vs D_IoU       | 0.811 |        0.0020 |
| EfficientNet-B0 | D_prob vs D_stability | 0.888 |        <0.001 |
| EfficientNet-B0 | D_prob vs D_faith     | 0.154 |        0.6331 |
| ResNet50        | D_prob vs D_SO        | 0.965 |        <0.001 |
| ResNet50        | D_prob vs D_IoU       | 0.965 |        <0.001 |
| ResNet50        | D_prob vs D_stability | 0.902 |        <0.001 |
| ResNet50        | D_prob vs D_faith     | 0.671 |        0.0202 |

The confidence-shift analysis shows stronger relationships with localization and explanation stability than the detector-performance analysis. These quantities remain distinct because a change in fake probability does not directly represent a change in classification performance.

The Approach 2 and Approach 3 analysis therefore provides two related but separate views:

* detector-performance degradation
* model-confidence and explanation degradation

---

## 14. Ordered severity analysis

Four transformation families have predefined severity orders:

* JPEG: Q80 → Q50 → Q20
* Resize: 0.75 → 0.50 → 0.25
* Darkening: 0.80 → 0.60 → 0.40
* Brightening: 1.20 → 1.40 → 1.60

Spearman correlation is used to test whether explanation metrics change monotonically with these predefined orders.

For Xception, ES shows a Spearman correlation of -1 across all four ordered transformation families. This indicates that explanation similarity decreases monotonically across the predefined severity sequence in each family.

For SO, Xception also shows a Spearman correlation of -1 for JPEG, resizing, and brightening. Darkening shows the opposite direction, with a correlation of +1, because SO increases across the tested darkening sequence.

These relationships describe the observed ordering within the evaluated severity conditions. They do not establish a causal mechanism for the changes.

---

## 15. Architecture comparison

The three architectures exhibit different explanation characteristics.

Xception has the highest clean SO and IoU among the three models:

* SO = 0.2948
* IoU = 0.1919

ResNet50 has slightly lower localization values:

* SO = 0.2584
* IoU = 0.1557

EfficientNet-B0 has lower clean localization values:

* SO = 0.1585
* IoU = 0.0915

However, these values alone do not establish an overall architecture ranking because explainability quality is multidimensional. Localization, faithfulness, stability, prediction behavior, and detector performance capture different properties.

The transformation results also show architecture-specific sensitivity. EfficientNet-B0 generally experiences larger explanation-stability reductions under resizing and brightness changes than Xception and ResNet50, while Xception and ResNet50 show particularly large stability reductions under strong JPEG compression.

---

## 16. Global explanation analysis

Global Grad-CAM maps are generated after normalizing individual maps to [0, 1] and resizing them to a common spatial resolution.

The aggregation includes:

* overall manipulated-image heatmaps
* Deepfakes heatmaps
* Face2Face heatmaps
* FaceSwap heatmaps
* NeuralTextures heatmaps

The global maps are used as descriptive summaries of the spatial regions emphasized by the model across the evaluated samples.

They are not treated as independent quantitative observations. Quantitative localization analysis is performed at the frame and aggregated source-video levels using the defined SO, IoU, saliency-mass, and hit-rate measures.

---

## 17. Statistical analysis

The statistical analysis is performed using the video-level or source-level unit appropriate to each analysis.

The main procedures are:

* Paired Wilcoxon signed-rank tests for clean-versus-transformed paired metrics
* Bootstrap 95% confidence intervals
* Rank-biserial effect sizes
* Spearman rank correlation for transformation relationships
* Benjamini-Hochberg false discovery rate correction

The source-level explainability analysis uses the 12 manipulated source-pair identifiers. The prediction-state analysis remains at the category-video level.

The Approach 2 and Approach 3 relationship analysis contains two predefined FDR families:

* Detector-performance family: 12 tests
* Confidence-shift family: 12 tests

Raw p-values are retained alongside BH-adjusted p-values.

Explanation stability is treated as a direct clean-to-transformed measurement. It is not subjected to a clean-versus-transformed Wilcoxon test because the clean reference has no transformed-pair stability value.

---

## 18. Validation and implementation safeguards

The final output structure was validated for all three evaluated models.

| Model           | Frame rows | Video rows | Source IDs | Valid ES_cos | Valid explanation IoU |
| --------------- | ---------: | ---------: | ---------: | -----------: | --------------------: |
| Xception        |     81,952 |        780 |         24 |          576 |                   576 |
| EfficientNet-B0 |     81,952 |        780 |         24 |          576 |                   576 |
| ResNet50        |     81,952 |        780 |         24 |          576 |                   576 |

Additional validation confirmed:

* 12 Original source-video IDs
* 12 manipulated source-pair IDs
* 60 manipulated category-video units
* no missing SO values for transformed fake observations
* complete transformed explanation-stability observations
* consistent source-video aggregation across models

Several implementation safeguards were added during the experiment.

The Grad-CAM cache uses category-aware identifiers to avoid collisions between manipulation categories. Clean prediction references also include category information.

The faithfulness implementation applies the Grad-CAM mask to the original-resolution transformed face crop rather than using a downsampled image as the baseline probability input.

Raw Grad-CAM maps are retained as float32 arrays for reproducibility.

Frame-level tensors and temporary arrays are released after use. Grad-CAM hooks and model references are explicitly cleaned up, and CUDA cache cleanup is performed at model boundaries.

These changes were made to reduce memory retention during the full multi-model evaluation.

---

## 19. Generated artifacts

The corrected experiment outputs are stored under:

```text
data/output/explainability_corrected_v1/
```

with model-specific directories:

```text
xception/seed_2024/
efficientnet_b0/seed_2024/
resnet50/seed_123/
```

Each model output contains:

```text
frame_level_results.csv
video_level_results.csv
statistics_results.csv
summary.json
```

and the generated global heatmaps.

The post-hoc analysis outputs are stored under:

```text
data/output/approach3_posthoc/
```

including:

```text
source_video_results.csv
explanation_stability.csv
paired_changes.csv
prediction_state_analysis.csv
severity_analysis.csv
localization_faithfulness.csv
approach2_explanation_integration.csv
approach2_explanation_relationships.csv
```

The main analysis scripts are:

```text
scripts/approach3_posthoc_analysis.py
scripts/validate_approach3_outputs.py
```

---

## 20. Limitations

Grad-CAM provides a model-derived visualization but does not establish that the highlighted region is causally responsible for the prediction.

The FaceForensics++ manipulation mask provides a reference for spatial localization, but agreement with the mask does not by itself prove that the model uses only manipulation-related evidence.

The source-level paired analysis contains 12 manipulated source-pair observations. This limits the effective sample size for correlation and paired statistical analysis.

The current robustness evaluation is restricted to the four predefined transformation families and their fixed severity levels. Noise, blur, contrast, and crop are not part of the current experiment.

The four manipulation categories originating from the same source-pair structure are not statistically independent. This is why source-level aggregation is used for the primary paired explainability analyses.

The observed correlations describe relationships between measured quantities. They do not establish causal relationships between detector performance, explanation stability, localization, and faithfulness.

---

## 21. Main empirical findings

The Approach 3 experiment provides the following empirical observations:

1. Clean Grad-CAM localization differs across architectures. Xception records SO = 0.2948, ResNet50 = 0.2584, and EfficientNet-B0 = 0.1585.

2. Localization agreement with the manipulation mask is not uniformly high, including for correctly classified manipulated images.

3. JPEG compression produces substantial reductions in both localization and explanation similarity, particularly at Q20.

4. Resizing produces smaller explanation changes for Xception and ResNet50 than JPEG compression, while EfficientNet-B0 shows greater sensitivity to reduced resolution.

5. Brightening and darkening produce architecture-specific explanation changes rather than a uniform response across models.

6. Prediction preservation and explanation preservation are related but distinct. An unchanged prediction can coexist with either high or low explanation similarity.

7. Explanation stability does not imply localization correctness. The Incorrect → Incorrect state for Xception shows high explanation similarity alongside low localization agreement.

8. Clean localization and blur-based faithfulness are not strongly correlated in the reported Xception analysis.

9. Detector-performance degradation and explanation-stability degradation show strong positive relationships for all three architectures after BH-FDR correction.

10. Fake-probability shifts show strong relationships with localization and explanation-stability changes, but these confidence changes are kept separate from detector-performance degradation.

11. The results support evaluating deepfake explanations using multiple dimensions rather than relying on a single explanation metric.

12. The combination of detector-performance degradation, localization change, faithfulness change, explanation stability, and prediction-state analysis provides a common framework for examining how CNN explanations behave under controlled input transformations.
