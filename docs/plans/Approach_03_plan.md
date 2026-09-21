# Approach 3 plan: explainability, faithfulness, and explanation robustness

## 1. Objective

Approach 3 evaluates whether the trained CNN detectors from Approach 1 produce explanations that are:

1. spatially aligned with manipulated regions;
2. faithful to the model's prediction;
3. stable under controlled transformations;
4. consistent across frames, videos, manipulation categories, and architectures;
5. related to the prediction-state changes measured in Approach 2.

The CNNs are not retrained.

The trained Approach 1 checkpoints are treated as fixed detectors.

The central research question is:

> **How accurately, consistently, and faithfully does the CNN explain its deepfake decisions, and how do these explanation properties change when the input is subjected to controlled transformations?**

---

# 2. Scope

## 2.1 Models

Run the core quantitative analysis for:

* Xception
* EfficientNet-B0
* ResNet50

Use the same test split as Approach 1.

Each architecture uses its own independently selected seed.

The best model can receive more detailed visual analysis and case studies, but the core quantitative analysis must cover all three architectures.

---

## 2.2 Seed selection

Before implementation, inspect the actual Approach 1 result artifacts.

Determine:

* exact run directory;
* result file;
* exact F1 field;
* exact ROC-AUC field;
* checkpoint path;
* model-to-seed relationship.

The selector must not guess filenames or field names.

The intended selection rule is:

1. highest video-level F1;
2. if tied, highest video-level ROC-AUC.

The selected checkpoint must be verified against:

* model;
* seed;
* split;
* source result artifact.

Record all of this in Approach 3 metadata.

Manual selection remains available:

```bash
python -m src.explainability --model xception --seed 2024
```

A manual seed overrides automatic selection but does not bypass checkpoint validation.

---

# 3. CLI behavior

Default:

```bash
python -m src.explainability
```

This runs the selected/default model using automatic seed selection.

Specific model:

```bash
python -m src.explainability --model xception
```

Specific seed:

```bash
python -m src.explainability --model xception --seed 2024
```

Robustness analysis:

```bash
python -m src.explainability --model xception --robustness
```

All models:

```bash
python -m src.explainability --check-all
```

`--check-all` must:

1. iterate over Xception, EfficientNet-B0, and ResNet50;
2. perform **independent automatic seed selection for each architecture**;
3. validate the selected checkpoint separately;
4. create independent output directories;
5. record the selected model and seed for each run.

Therefore:

```bash
python -m src.explainability --check-all
```

must **not** interpret a globally supplied seed as the seed for every architecture.

If `--check-all --seed 42` is supplied, the CLI must reject the combination with a clear error message rather than silently applying seed 42 to all models.

---

# 4. Data

Use the same FaceForensics++ C23 test data and split used by Approach 1 and Approach 2.

The generated manifest and existing experiment artifacts are the source of truth for the actual test set.

Do not silently regenerate a different split.

Quantitative manipulation-mask analysis is performed only on:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

Original images are excluded from quantitative mask-based localization metrics.

---

# 5. Face preprocessing & bounding box tracking

Reuse the existing face preprocessing.

The pipeline:

```text
full frame
   |
   v
MTCNN
   |
   v
selected face
   |
   v
tracking
   |
   v
30% expanded bounding box
   |
   v
face crop
```

Approach 3 must use the same face crop.

Do not introduce another face detector or crop strategy.

**Bounding box verification:** During Phase 1 inspection, verify whether the expanded bounding box coordinates `[x1, y1, x2, y2]` are stored in `manifest.csv` or an associated metadata artifact. If stored, load them directly. If bounding boxes were not persisted during Approach 1 preprocessing, document the exact extraction/matching method before implementing mask mapping.

---

# 6. Manipulation-mask alignment

For every manipulated frame:

1. identify category;
2. identify video;
3. identify `original_frame_number`;
4. locate the corresponding FF++ mask;
5. transform the full-frame mask into the existing face-crop coordinate system.

Do not assume that processed frame index equals the FF++ mask frame index.

Validate representative examples from:

* Deepfakes;
* Face2Face;
* FaceSwap;
* NeuralTextures.

Validation must confirm:

* correct frame;
* correct manipulation mask;
* correct face crop;
* correct mask-to-face-crop alignment.

---

# 7. Resize transformation mask handling

This is explicitly defined from the actual `src/robustness.py` implementation.

Approach 2 `apply_resize()` performs:

```text
face crop
    |
    v
downsample
    |
    v
upsample back to original H × W
```

The returned transformed image therefore has exactly the same height and width as the original face crop.

Consequently:

> **The manipulation mask remains unchanged for the resize conditions.**

The mask should **not** be independently downsampled and re-binarized.

This avoids introducing artificial boundary changes that are not present in the actual transformed image coordinate system.

The same face-crop manipulation mask is used for:

* clean;
* JPEG;
* all resize conditions;
* darkening;
* brightening.

The current enabled transformations do not change the face-crop spatial dimensions.

If future transformations such as cropping are enabled, mask transformation must be reconsidered separately.

---

# 8. Grad-CAM

Use `pytorch-grad-cam` after verifying compatibility with:

* PyTorch;
* timm;
* CUDA/runtime.

Pin the tested version in `requirements.txt`.

Record the installed version in run metadata.

---

## 8.1 Target

Always use the fake-class logit.

This applies regardless of prediction:

```text
GT fake + predicted fake
    -> fake-class Grad-CAM

GT fake + predicted real
    -> fake-class Grad-CAM
```

---

## 8.2 Target layer

Use an architecture-aware resolver for the final suitable convolutional feature layer.

The actual layer must be verified against the instantiated `timm` model.

Store the selected layer in the run metadata.

---

## 8.3 Gradient context & memory management

Prediction:

```python
model.eval()

with torch.inference_mode():
    ...
```

Grad-CAM:

```python
model.eval()

# gradients enabled
GradCAM(...)
```

Never wrap Grad-CAM in:

```python
torch.no_grad()
```

or:

```python
torch.inference_mode()
```

**Memory management:**
* Initial Grad-CAM batch size is 1.
* Explicitly detach and move tensors to CPU (`.detach().cpu().numpy()`) immediately after extraction.
* Explicitly delete intermediate tensors and call `torch.cuda.empty_cache()` periodically (e.g., between video groups) to prevent memory fragmentation on the GPU.
* Only optimize to larger batches after verifying that batch-size-1 and larger-batch outputs agree for the same inputs.

---

# 9. Grad-CAM resolution

The Grad-CAM map is generated at feature-map resolution.

Upsample it to the original face-crop resolution before:

* thresholding;
* SO;
* IoU;
* saliency mass;
* visualization;
* explanation stability.

Clean and transformed maps must have identical spatial dimensions before comparison.

Implement this as a hard validation.

If:

```python
clean_map.shape != transformed_map.shape
```

the corresponding frame must be recorded as failed, excluded from that metric, and processing should continue.

The failure must not be silently ignored.

---

# 10. Local explanation

For every manipulated test frame:

```text
face crop
   |
   v
CNN prediction
   |
   v
fake probability
   |
   v
fake-class Grad-CAM
   |
   v
normalized Grad-CAM
   |
   v
upsample to face-crop resolution
   |
   +--> localization
   |
   +--> faithfulness
```

Store the frame-level result before aggregation.

---

# 11. Saliency thresholds

Primary threshold:

> Top 20% Grad-CAM activation.

Also calculate sensitivity results at:

* top 10%;
* top 20%;
* top 30%.

The 20% threshold is fixed before experimentation.

No threshold is tuned against the test set.

---

# 12. Localization metrics

## 12.1 Saliency Overlap

$$
SO=\frac{|S\cap M|}{|S|}
$$

where:

* `S` is the binary salient region;
* `M` is the manipulation mask.

---

## 12.2 Regional IoU

$$
IoU=\frac{|S\cap M|}{|S\cup M|}
$$

---

## 12.3 Saliency mass inside manipulation mask

$$
SM_{mask}
=
\frac{
\sum_{p\in M}G(p)
}{
\sum_pG(p)
}
$$

---

## 12.4 Maximum-activation mask hit rate

Let:

$$
p^*=\arg\max_pG(p)
$$

Then:

$$
Hit =
\begin{cases}
1 & p^*\in M\\
0 & p^*\notin M
\end{cases}
$$

---

# 13. Video-level aggregation

Frames are not treated as independent statistical observations.

For each video, aggregate valid frame-level results.

Primary aggregation:

* mean.

Also retain:

* median;
* number of valid frames;
* number of failed frames.

The video is the statistical unit for inferential analysis.

---

# 14. Global explanation analysis

Compute all candidate global metrics before selecting a primary global metric.

Candidates:

1. aggregated SO;
2. aggregated IoU;
3. saliency mass;
4. maximum-activation mask hit rate;
5. within-video explanation consistency;
6. within-category explanation consistency;
7. saliency entropy.

The final report may select a primary global metric using predefined criteria:

* interpretability;
* stability;
* non-redundancy;
* empirical variation.

No global metric should be declared primary in the implementation merely because it produces favorable results.

---

# 15. Global explanation consistency

## 15.1 Within-video

Normalize each valid frame Grad-CAM.

Resize to a common spatial resolution.

Calculate:

```text
frame maps
    |
    v
mean video map
```

Then calculate cosine similarity between each frame map and the video mean map.

---

## 15.2 Within-category

Calculate mean video-level maps.

Then:

```text
video maps
    |
    v
category mean map
```

Calculate cosine similarity between each video map and its category mean.

These are supporting global consistency metrics and are separate from clean-versus-transformed explanation stability.

---

# 16. Saliency entropy

Calculate saliency entropy as a descriptive supporting metric.

Do not automatically interpret higher or lower entropy as better.

Retain it in machine-readable output.

It may be moved to supplementary analysis in the final report if it does not provide useful non-redundant information.

---

# 17. Global visual explanation

Generate mean Grad-CAM heatmaps for:

* all manipulated images;
* Deepfakes;
* Face2Face;
* FaceSwap;
* NeuralTextures.

The heatmaps are visual support for the quantitative analysis.

---

# 18. Correct versus missed fake

Define:

### Correct fake

```text
GT = fake
prediction = fake
```

### Missed fake

```text
GT = fake
prediction = real
```

Compare:

* SO;
* IoU;
* saliency mass;
* hit rate;
* faithfulness;
* explanation stability.

Visual examples should be selected using predefined quantitative rules.

---

# 19. Faithfulness

Primary metric:

$$
\Delta P_{fake}
=
P_{fake}(x)-P_{fake}(x_{masked})
$$

The salient region is the top 20% Grad-CAM region.

Default masking:

> blur masking.

Sensitivity masking methods:

* zero masking;
* mean-value masking.

---

# 20. Faithfulness on transformed inputs

For each transformation `t`:

```text
clean image x
    |
    v
transformation t
    |
    v
x_t
    |
    +--> P_fake(x_t)
    |
    +--> Grad-CAM G_t
             |
             v
        top 20% S_t
             |
             v
       mask S_t in x_t
             |
             v
       P_fake(x_t,masked)
```

Calculate:

$$
\Delta P_{fake,t}
=
P_{fake}(x_t)
-
P_{fake}(x_{t,masked})
$$

Do not transform the masked image again.

This prevents double application of the corruption.

For clean:

$$
\Delta P_{fake,clean}
=
P_{fake}(x)
-
P_{fake}(x_{clean,masked})
$$

Then:

$$
D_{faith}
=
\Delta P_{fake,clean}
-
\Delta P_{fake,t}
$$

---

# 21. Explanation stability

For every clean/transformed frame pair, calculate explanation stability regardless of whether prediction changes.

Primary continuous metric:

$$
ES_{cos}
=
\frac{G_c\cdot G_t}
{||G_c||||G_t||}
$$

Secondary region-level metric:

$$
IoU_{exp}
=
\frac{|S_c\cap S_t|}
{|S_c\cup S_t|}
$$

Calculate both at:

* 10%;
* 20%;
* 30%.

The 20% threshold remains primary.

---

# 22. Prediction-preserved versus prediction-changed

Define:

```text
prediction preserved:
pred_clean == pred_transformed

prediction changed:
pred_clean != pred_transformed
```

Compare:

* `ES_cos`;
* explanation IoU;
* SO;
* IoU;
* faithfulness.

This directly evaluates whether prediction preservation corresponds to explanation preservation.

No such equivalence should be assumed.

---

# 23. Four-state explanation analysis

Use:

| Prediction | Explanation |
| ---------- | ----------- |
| Preserved  | Stable      |
| Preserved  | Changed     |
| Changed    | Stable      |
| Changed    | Changed     |

For descriptive binary grouping:

```text
ES_cos >= 0.90 -> explanation stable
ES_cos < 0.90  -> explanation changed
```

This threshold is fixed before the experiment.

The continuous `ES_cos` remains the primary stability measure.

---

# 24. Robustness transformations

Use all 12 Approach 2 conditions:

### JPEG

* Q80
* Q50
* Q20

### Resize

* 0.75
* 0.50
* 0.25

### Darkening

* 0.80
* 0.60
* 0.40

### Brightening

* 1.20
* 1.40
* 1.60

Reuse the existing Approach 2 transformation implementation.

---

# 25. Transformation-mask rules

The current transformation implementation returns images with the original face-crop dimensions.

Therefore:

| Transformation | Mask handling        |
| -------------- | -------------------- |
| Clean          | clean face-crop mask |
| JPEG           | same mask            |
| Resize         | same mask            |
| Darkening      | same mask            |
| Brightening    | same mask            |

For resize specifically:

```text
image:
face crop -> downsample -> upsample -> same H × W

mask:
face-crop mask -> unchanged
```

No bilinear mask interpolation or 0.5 re-binarization is required.

This conclusion is based on the current `src/robustness.py` implementation.

If future geometry-changing transformations are enabled, their mask transformation must be specified separately before implementation.

---

# 26. Transformation families

Group results into:

* JPEG;
* Resize;
* Darkening;
* Brightening.

Report individual conditions and grouped severity behavior.

---

# 27. Severity analysis

Treat severity as ordered within each family:

```text
JPEG:
Q80 -> Q50 -> Q20

Resize:
0.75 -> 0.50 -> 0.25

Darkening:
0.80 -> 0.60 -> 0.40

Brightening:
1.20 -> 1.40 -> 1.60
```

Do not assume monotonic degradation.

Test the observed relationship.

---

# 28. Approach 2 integration

Use:

$$
D_{F1}
=
F1_{clean}
-
F1_{transformed}
$$

For explanation:

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
1-ES_{cos}
$$

$$
D_{faith}
=
\Delta P_{fake,clean}
-
\Delta P_{fake,transformed}
$$

All of these are evaluated at the video level.

---

# 29. Prediction-state analysis

Reuse Approach 2's four states:

1. Correct -> Correct
2. Correct -> Incorrect
3. Incorrect -> Correct
4. Incorrect -> Incorrect

Attach Approach 3 measurements to each state:

* SO;
* IoU;
* saliency mass;
* hit rate;
* `ES_cos`;
* explanation IoU;
* faithfulness.

The prediction-state definitions must not be independently redefined in Approach 3.

---

# 30. Detector degradation versus explanation degradation

Compare:

$$
D_{F1}
$$

against:

$$
D_{SO}
$$

$$
D_{IoU}
$$

$$
D_{stability}
$$

and:

$$
D_{faith}
$$

Use video-level paired observations.

The objective is to determine whether detector degradation and explanation degradation behave similarly.

Do not interpret a correlation as evidence of causation.

---

# 31. Localization versus faithfulness

Calculate Spearman correlations:

$$
SO \leftrightarrow \Delta P_{fake}
$$

$$
IoU \leftrightarrow \Delta P_{fake}
$$

$$
SM_{mask} \leftrightarrow \Delta P_{fake}
$$

Analyze:

* overall;
* by architecture;
* by manipulation category;
* correct fake;
* missed fake;
* clean/transformed.

---

# 32. Architecture analysis

Run the core quantitative analyses for:

* Xception;
* EfficientNet-B0;
* ResNet50.

Compare architecture-specific:

* localization;
* faithfulness;
* explanation stability;
* global consistency;
* transformation behavior;
* prediction-state behavior.

Do not produce an overall architecture ranking.

---

# 33. Statistical unit

The video is the statistical unit.

The workflow is:

```text
frame-level results
       |
       v
video-level aggregation
       |
       v
statistical testing
```

Frames remain useful for visualization and descriptive analysis.

---

# 34. Statistical tests

Primary paired test:

> Wilcoxon signed-rank test.

Effect size:

> rank-biserial correlation.

Confidence interval:

> video-level bootstrap 95% CI.

Correlation:

> Spearman rank correlation.

---

# 35. Small-sample gate

Wilcoxon:

> require at least 6 non-zero paired differences.

Spearman:

> require at least 6 paired observations.

Bootstrap:

> execute only when the configured minimum sample requirement is satisfied.

Otherwise:

```text
test_status = not_tested_small_n
```

Do not report an artificial p-value.

Every statistical result should contain:

```text
n_total
n_valid
n_nonzero
test_status
```

---

# 36. FDR correction

Use Benjamini-Hochberg FDR correction.

The correction families are explicitly separated.

### Family 1: localization

* SO;
* IoU;
* saliency mass;
* hit rate.

### Family 2: faithfulness

* blur masking;
* zero masking;
* mean-value masking.

### Family 3: explanation stability

* `ES_cos`;
* explanation IoU.

### Family 4a: localization-faithfulness relationships

Spearman correlations between:

* SO and faithfulness;
* IoU and faithfulness;
* saliency mass and faithfulness.

### Family 4b: detector-explanation degradation relationships

Spearman correlations between:

* detector degradation and SO degradation;
* detector degradation and IoU degradation;
* detector degradation and explanation stability degradation;
* detector degradation and faithfulness degradation.

These families must not be combined.

The FDR family definitions must be recorded in the statistical output.

---

# 37. Master video-level table

Create:

```text
video_level_results.csv
```

Stable keys:

```text
model
seed
video_id
transformation
```

Include:

```text
category
ground_truth

clean_prediction
transformed_prediction

clean_prob_fake
transformed_prob_fake

prediction_state
prediction_preserved

SO
IoU
saliency_mass
hit_rate

faithfulness_blur
faithfulness_zero
faithfulness_mean

ES_cos
explanation_IoU

transformation_family
severity

n_valid_frames
n_failed_frames
```

Convenience CSVs can be derived from this master table.

---

# 38. Grad-CAM storage and caching

Raw normalized Grad-CAM maps must be stored as:

> **one float32 NumPy `.npy` file per frame.**

Storage optimization considerations:
* Native input resolution (e.g., $299 \times 299$ for Xception, $224 \times 224$ for ResNet/EfficientNet) should be preferred for raw caching to conserve disk space.
* Total footprint across 3 models × 13 conditions × ~3,152 manipulated frames is estimated at ~25–35 GB. The pilot experiment (§47) must measure the exact per-frame `.npy` byte size on disk before full execution.

Recommended structure:

```text
data/output/explainability/
    model/
        seed_XXX/
            cache/
                gradcam/
                    clean/
                    jpeg_q80/
                    jpeg_q50/
                    ...
```

The cache key must encode enough information to prevent accidental reuse across:

* models;
* seeds;
* transformations;
* frames.

The associated metadata must identify:

* model;
* seed;
* video;
* frame;
* transformation;
* target layer;
* Grad-CAM method;
* map resolution.

---

# 39. Raw versus derived artifacts

## Raw reusable artifacts

* transformed face crop;
* fake probability;
* prediction;
* normalized Grad-CAM;
* manipulation mask;
* checkpoint metadata.

## Derived artifacts

* binary salient region;
* SO;
* IoU;
* saliency mass;
* hit rate;
* faithfulness;
* explanation IoU;
* statistical tests;
* visual case selection.

Changing the threshold must reuse the stored Grad-CAM.

Changing statistical analysis must not regenerate Grad-CAM.

---

# 40. Resume behavior

On restart:

1. inspect existing artifact;
2. validate metadata;
3. determine whether it is reusable;
4. skip valid raw computation;
5. recompute invalid/incomplete artifacts.

Do not treat file existence alone as proof that an artifact is valid.

---

# 41. Global output structure

Recommended:

```text
data/
└── output/
    └── explainability/
        ├── xception/
        │   └── seed_XXX/
        │       ├── frame_level_results.csv
        │       ├── video_level_results.csv
        │       ├── summary.json
        │       ├── category_metrics.csv
        │       ├── global_metrics.csv
        │       ├── cache/
        │       │   └── gradcam/
        │       ├── statistics/
        │       ├── global_heatmap.png
        │       ├── correct/
        │       ├── incorrect/
        │       └── robustness/
        │
        ├── efficientnet_b0/
        │   └── seed_XXX/
        │
        └── resnet50/
            └── seed_XXX/
```

Each architecture receives its own independently selected seed directory under `--check-all`.

---

# 42. Representative visual examples

Automatically select a fixed number of examples for:

* correct fake;
* missed fake;
* strongest SO;
* weakest SO;
* prediction-preserved transformation;
* prediction-changed transformation.

Selection must be based on predefined quantitative rules.

Do not manually cherry-pick examples after seeing the results.

---

# 43. Visual outputs

For each selected example save:

1. original face crop;
2. Grad-CAM heatmap;
3. Grad-CAM overlay;
4. Grad-CAM + manipulation-mask overlay;
5. binary salient region where useful.

For robustness examples:

```text
clean image
transformed image
clean Grad-CAM
transformed Grad-CAM
```

---

# 44. Reproducibility metadata

Every run should record:

```text
model
seed
checkpoint path
Approach 1 selection artifact
selected F1
selected ROC-AUC

target layer

test split
manifest identifier

PyTorch version
timm version
pytorch-grad-cam version
CUDA/runtime

transformation configuration
threshold configuration
masking configuration

statistical-test configuration
bootstrap configuration
FDR family configuration
minimum-n configuration
```

---

# 45. Quality-control gates

## Gate 1: environment

Verify:

* dependency versions;
* Grad-CAM compatibility;
* runtime;
* GPU.

## Gate 2: checkpoint selection

Verify:

* model;
* seed;
* checkpoint;
* split;
* result artifact.

## Gate 3: Grad-CAM

Verify:

* fake-class target;
* correct target layer;
* gradients enabled;
* finite maps;
* correct resolution.

## Gate 4: mask

Verify:

* exact frame;
* category;
* face crop;
* mask alignment.

For resize, verify that the unchanged face-crop mask corresponds to the transformed image coordinate system.

## Gate 5: prediction reproduction

Approach 3 clean predictions must reproduce Approach 1/2 predictions within an explicitly documented tolerance.

## Gate 6: clean explainability

Verify:

* successful Grad-CAM count;
* failure count;
* localization;
* faithfulness;
* video aggregation.

## Gate 7: transformed explainability

Verify:

* all 12 transformations;
* transformed predictions;
* transformed Grad-CAM;
* mask alignment;
* faithfulness;
* stability.

## Gate 8: statistical analysis

Verify:

* video-level unit;
* minimum-n rules;
* FDR family assignment;
* traceability.

## Gate 9: reproducibility

Verify:

* rerun consistency;
* raw-map caching;
* threshold changes reuse Grad-CAM;
* statistical changes do not rerun explanations.

---

# 46. Implementation order

### Phase 1: environment and repository inspection

1. inspect requirements;
2. inspect runtime versions;
3. pin Grad-CAM;
4. inspect Approach 1 artifacts;
5. establish seed-selection logic;
6. inspect and reuse Approach 2 transformation behavior;
7. verify bounding box persistence for FF++ mask mapping.

### Phase 2: model and Grad-CAM

1. checkpoint loading;
2. checkpoint validation;
3. target-layer resolver;
4. fake-class target;
5. Grad-CAM;
6. batch-size-1 validation;
7. float32 `.npy` caching.

### Phase 3: mask pipeline

1. mask lookup;
2. frame matching;
3. face-crop mapping;
4. validation across all four manipulation categories;
5. explicit resize-mask validation.

### Phase 4: clean explainability

1. predictions;
2. Grad-CAM;
3. localization;
4. faithfulness;
5. video aggregation;
6. global metrics.

### Phase 5: robustness

1. Approach 2 transformations;
2. transformed predictions;
3. transformed Grad-CAM;
4. transformed localization;
5. transformed faithfulness;
6. explanation stability;
7. prediction-state linkage.

### Phase 6: statistics

1. paired comparisons;
2. effect sizes;
3. bootstrap;
4. Spearman correlations;
5. separate FDR families;
6. small-sample handling.

### Phase 7: visual analysis

1. global heatmaps;
2. category heatmaps;
3. correct/missed cases;
4. robustness cases;
5. prediction/explanation four-state examples.

---

# 47. Pilot experiment

Before the complete experiment, run:

```text
1 architecture
1 selected seed
1-2 videos per manipulation category

clean
JPEG Q20
resize 0.25
darkening 0.40
brightening 1.60
```

Validate:

* checkpoint;
* prediction reproduction;
* mask alignment;
* Grad-CAM;
* SO;
* IoU;
* faithfulness;
* stability;
* output structure;
* cache reuse;
* resize mask behavior;
* disk usage per frame `.npy`.

Only after the pilot passes all gates should the full experiment begin.

---

# 48. Full experiment

The full experiment consists of:

```text
3 architectures
+
independently selected seed per architecture
+
all manipulated test frames
+
clean condition
+
12 transformations
+
3 saliency thresholds
+
3 masking methods
```

The expensive raw Grad-CAM computation should be performed once and reused for derived analyses.

---

# 49. Research questions answered

The final results should provide evidence for:

1. How well does Grad-CAM localize manipulated regions?
2. How faithful are highlighted regions to fake predictions?
3. Are explanations stable under controlled transformations?
4. Does explanation behavior change with transformation severity?
5. Do explanation properties differ across manipulation categories?
6. Do correct and missed fake predictions have different explanations?
7. Do the three CNN architectures exhibit different explanation behavior?
8. Does detector degradation correspond to explanation degradation?
9. Does prediction preservation correspond to explanation preservation?
10. Is spatial localization associated with intervention-based faithfulness?

---

# 50. Novelty framing

Do not claim novelty for individual established techniques such as:

* Grad-CAM;
* SO;
* IoU;
* masking-based faithfulness;
* explanation stability;
* robustness testing.

The contribution should be framed around their **integrated, transformation-conditioned evaluation** under a common protocol.

The strongest analysis dimensions are:

### Transformation-conditioned explanation reliability

Jointly analyze:

```text
transformation
+
severity
+
prediction
+
localization
+
faithfulness
+
explanation stability
```

### Localization versus faithfulness

Test whether spatial agreement with manipulation masks is associated with intervention-based faithfulness.

### Prediction-state-conditioned explanation analysis

Connect Approach 2's four prediction states with Approach 3 explanation properties.

### Detector versus explanation degradation

Compare changes in detector performance against changes in explanation behavior.

### Prediction-preserving transformations

Test whether unchanged predictions necessarily correspond to unchanged explanations.

The final novelty claim should depend on the actual results and literature review. No individual metric should be presented as novel.

---

# 51. Methodological boundaries

Do not assume that:

* localization implies faithfulness;
* faithfulness implies localization;
* prediction preservation implies explanation preservation;
* explanation preservation implies prediction preservation;
* detector robustness implies explanation robustness;
* detector degradation causes explanation degradation;
* correlation establishes causation.

Approach 3 measures these relationships empirically.

The main methodological contribution is the controlled integration of:

```text
detector robustness
+
prediction state
+
explanation localization
+
faithfulness
+
explanation stability
```

under the same transformation protocol.
