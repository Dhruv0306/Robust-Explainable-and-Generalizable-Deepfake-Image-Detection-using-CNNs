# Approach 4 - Generalizability to unseen manipulation techniques

## 1. Objective

Approach 4 evaluates the generalizability of CNN-based deepfake detectors when a manipulation technique is excluded from the training distribution.

The primary research question is:

> **To what extent does a CNN trained on multiple manipulation categories generalize to a manipulation category excluded from training?**

The experiment compares two conditions for each manipulation \(M\):

1. **Full-training condition:** \(M\) is represented during training.
2. **Leave-one-manipulation-out condition:** \(M\) is completely excluded from training and validation.

Both models are evaluated on the same balanced test condition:

$$
Original + M
$$

This produces a controlled comparison between performance when the manipulation is seen during training and performance when it is unseen during training.

The central effect is:

$$
\boxed{
\Delta Metric_M =
Metric_{Full,M} - Metric_{LOO,M}
}
$$

where the primary metrics are:

* F1-score
* ROC-AUC
* Fake-class recall

A positive \(\Delta\) means the full-training model performed better on the held-out manipulation.

A value close to zero indicates little measurable difference.

A negative value means the LOO model performed better on that manipulation. This is a valid empirical result and must not be treated as an implementation failure.

---

# 2. Scope of generalization

Approach 4 distinguishes three separate forms of generalization.

### 2.1 Intra-dataset manipulation generalization

The model is trained on multiple FF++ manipulation categories and evaluated on an FF++ manipulation category excluded from training.

```text
FaceForensics++
      │
      ├── training: 3 manipulation categories
      │
      └── testing: 1 unseen manipulation category
```

This is the **primary Approach 4 experiment**.

### 2.2 Cross-dataset generalization

A model trained on the full FF++ training distribution is evaluated on Celeb-DF v2 without retraining or adaptation.

```text
FaceForensics++
      ↓
Frozen model
      ↓
Celeb-DF v2
```

This is a separate experiment and should not be interpreted as pure manipulation generalization.

### 2.3 Explainability under distribution shift

The existing Grad-CAM implementation is used to compare prediction-associated spatial evidence for seen and unseen manipulation conditions.

```text
LOO model
    ↓
seen manipulation / unseen manipulation
    ↓
Grad-CAM
```

This is secondary supporting evidence for the generalization analysis.

---

# 3. Dataset

The experiment uses the project's existing **650-video FaceForensics++ C23 subset**.

| Category       |  Videos |
| -------------- | ------: |
| Original       |     130 |
| Deepfakes      |     130 |
| Face2Face      |     130 |
| FaceSwap       |     130 |
| NeuralTextures |     130 |
| **Total**      | **650** |

The 650 videos represent the project-selected subset of the larger FaceForensics++ dataset.

The experiment does not claim that the results represent the complete official FaceForensics++ dataset.

The existing project preprocessing pipeline is retained.

---

# 4. Manipulation categories

The four manipulation categories are:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

Each category is held out independently.

Therefore, four LOO conditions are created:

$$
M \in
\{
Deepfakes,\ Face2Face,\ FaceSwap,\ NeuralTextures
\}
$$

---

# 5. Leakage-safe split

Approach 4 retains the split mechanism established in Approach 1.

The split is based on relationships between videos rather than independently assigning individual videos.

The mechanism is:

```text
Video relationships
        ↓
Relationship graph
        ↓
Connected components
        ↓
Component-level split assignment
        ↓
Train / validation / test
```

Entire connected components remain within a single split.

This prevents related videos from being distributed across train, validation, and test sets.

The existing generated `data/splits/splits.json` must be audited before implementation.

The audit must verify:

* train/validation/test membership
* video counts
* category counts
* connected-component membership
* train/validation/test overlap
* relationship-group overlap
* manipulation distribution
* whether the actual generated split is consistent with the intended leakage-control mechanism

The actual generated split is authoritative.

The configured target group counts in source code must not be assumed to represent the actual generated split without checking `splits.json`.

---

# 6. Reuse of the existing split

If the existing split passes the audit, Approach 4 will reuse it.

No new random split will be generated.

For LOO experiments, the existing split is filtered by manipulation category.

For held-out manipulation \(M\):

```text
Original
+
three non-M manipulations
```

remain available for training and validation.

The held-out \(M\) category is removed from both.

The test set retains the corresponding:

```text
Original + M
```

condition.

This means the split assignment itself remains consistent across the four LOO experiments.

---

# 7. Primary experimental comparison

For each manipulation \(M\), two models are compared.

### Full model

The model is trained using:

```text
Original
+ Deepfakes
+ Face2Face
+ FaceSwap
+ NeuralTextures
```

The corresponding Approach 1 checkpoint is reused if it passes the checkpoint audit.

### LOO model

The model is trained using:

```text
Original
+ all manipulations except M
```

The held-out \(M\) category is absent from:

* training
* validation
* checkpoint selection
* hyperparameter selection

The model is evaluated on:

```text
Original + M
```

---

# 8. Balanced per-manipulation evaluation

The primary comparison is always made on:

$$
Original + M
$$

rather than comparing pooled seen and unseen test sets.

For the expected 130-video category counts, each manipulation-specific evaluation contains:

```text
130 Original
130 Manipulated
```

giving a balanced binary test condition.

For example:

```text
Full FF++ model
       ↓
Original + NeuralTextures

LOO-NeuralTextures model
       ↓
Original + NeuralTextures
```

Both models therefore encounter the same test population.

This avoids the class-prevalence difference that would arise from comparing:

```text
Original + 3 seen manipulations
```

against:

```text
Original + 1 unseen manipulation
```

The pooled seen-versus-unseen comparison is not a primary experiment.

---

# 9. Full-training baseline

Approach 1 provides the full-training baseline.

The existing valid Approach 1 models are audited before reuse.

There are:

$$
3\ architectures \times 3\ seeds = 9
$$

full-training models.

The architectures are:

* Xception
* EfficientNet-B0
* ResNet50

The seeds are:

* 42
* 123
* 2024

The full models are evaluated separately on each manipulation-specific test condition.

Therefore the same full model can provide:

```text
Full → Deepfakes
Full → Face2Face
Full → FaceSwap
Full → NeuralTextures
```

No new full-model training is required if all nine existing checkpoints are valid.

---

# 10. Leave-one-manipulation-out training

Four independent LOO experiments are performed.

| LOO experiment     | Training distribution                             | Held-out test manipulation |
| ------------------ | ------------------------------------------------- | -------------------------- |
| LOO-Deepfakes      | Original + F2F + FaceSwap + NeuralTextures        | Deepfakes                  |
| LOO-Face2Face      | Original + Deepfakes + FaceSwap + NeuralTextures  | Face2Face                  |
| LOO-FaceSwap       | Original + Deepfakes + Face2Face + NeuralTextures | FaceSwap                   |
| LOO-NeuralTextures | Original + Deepfakes + Face2Face + FaceSwap       | NeuralTextures             |

For each LOO condition:

$$
3\ architectures \times 3\ seeds = 9
$$

training runs.

Across four held-out categories:

$$
4\times3\times3 = \boxed{36}
$$

new LOO training runs.

This is the expected new training workload after the existing checkpoint audit.

---

# 11. Architecture configuration

The three CNN architectures from Approach 1 are retained:

* Xception
* EfficientNet-B0
* ResNet50

All use ImageNet-pretrained initialization.

No architecture is selected using Approach 4 test performance.

No architecture-specific hyperparameter tuning is introduced.

---

# 12. Seed configuration

Approach 4 uses all three predefined seeds:

$$
\boxed{42,\ 123,\ 2024}
$$

This is a deliberate methodological change from using the test-selected Approach 3 seeds.

The Approach 3 seed-selection mechanism must not be used to select the Approach 4 seed.

For every architecture and LOO condition, all three seeds are trained.

The three seeds measure sensitivity to training stochasticity and initialization.

---

# 13. Seed variability versus bootstrap uncertainty

These are separate analyses and must remain separate throughout the implementation.

### Seed-level variability

For a given architecture and manipulation:

```text
Seed 42
Seed 123
Seed 2024
      ↓
mean ± standard deviation
```

This measures variation across independent training seeds.

### Video-level bootstrap uncertainty

For a fixed model comparison:

```text
Full model predictions
          │
          │
LOO model predictions
          │
          ↓
paired video-level predictions
          ↓
bootstrap over videos
          ↓
95% confidence interval
```

This measures uncertainty associated with the finite set of evaluated test videos.

The bootstrap must **not** treat the three seeds as though they were additional independent videos.

The implementation must therefore never:

1. calculate three seed-level F1 values,
2. average them,
3. bootstrap those three values as if they were video observations.

Instead, the bootstrap operates directly on the video-level predictions.

---

# 14. Training configuration

LOO training uses the same training configuration as Approach 1.

| Parameter                   | Configuration                              |
| --------------------------- | ------------------------------------------ |
| Initialization              | ImageNet pretrained                        |
| Optimizer                   | AdamW                                      |
| Learning rate               | \(1\times10^{-4}\)                         |
| Weight decay                | \(1\times10^{-4}\)                         |
| Loss                        | BCEWithLogitsLoss                          |
| Class weighting             | Recomputed from filtered LOO training data |
| Maximum epochs              | 30                                         |
| Early stopping patience     | 5                                          |
| Scheduler                   | ReduceLROnPlateau                          |
| Scheduler patience          | 2                                          |
| Frame sampling              | Every 4th frame                            |
| Effective sampling          | 7.5 FPS                                    |
| Face detector               | MTCNN                                      |
| IoU tracking threshold      | 0.5                                        |
| Bounding-box margin         | 0.3                                        |
| Minimum usable frames/video | 20                                         |
| Primary aggregation         | Mean                                       |
| Classification threshold    | 0.5                                        |

The LOO training configuration should not be modified to improve performance on the unseen manipulation.

---

# 15. Class-weight calculation

Class weights must be recomputed for every LOO training condition.

For example, LOO-NeuralTextures calculates the class weighting from:

```text
Original
Deepfakes
Face2Face
FaceSwap
```

only.

The NeuralTextures frames must not contribute to the training class-weight calculation.

This ensures that the loss function reflects the actual LOO training distribution.

Full-model class weights remain those used by the existing Approach 1 training pipeline.

---

# 16. Preprocessing

The same preprocessing pipeline established in Approach 1 is retained.

```text
Video
  ↓
Sample every 4th frame
  ↓
MTCNN face detection
  ↓
IoU-based temporal tracking
  ↓
Bounding-box expansion
  ↓
Face crop
  ↓
Architecture-specific resize
  ↓
ImageNet normalization
  ↓
CNN
```

Input sizes remain:

* Xception: 299 × 299
* EfficientNet-B0: 224 × 224
* ResNet50: 224 × 224

Existing face crops should be reused where they correspond to the same preprocessing configuration.

The LOO experiment should not introduce a separate preprocessing pipeline.

---

# 17. Checkpoint selection

Checkpoint selection is based exclusively on validation loss.

For each training run:

```text
Training
   ↓
Validation loss
   ↓
Lowest validation loss
   ↓
Best checkpoint
   ↓
Freeze checkpoint
   ↓
Test
```

The test set is never used to select the checkpoint.

The unseen manipulation must not influence checkpoint selection.

Early stopping also remains based on validation loss.

The checkpoint metadata should preserve at least:

* epoch
* model state
* optimizer state where applicable
* validation loss
* validation accuracy

---

# 18. Distinguishing checkpoint selection from seed selection

These are three different concepts:

| Concept              | Approach 4 rule            |
| -------------------- | -------------------------- |
| Seed                 | 42, 123, 2024              |
| Checkpoint selection | Lowest validation loss     |
| Seed selection       | No test-informed selection |

The implementation must not combine these concepts.

In particular, the Approach 3 `seed_selector.py` logic must not be called to select the best Approach 4 seed.

---

# 19. Video-level probability aggregation

Frames are not the primary statistical unit.

For every video, the model generates frame-level probabilities:

$$
p_1,p_2,\ldots,p_N
$$

These are aggregated using the existing primary method:

$$
\boxed{
p_{video}
=
\frac{1}{N}\sum_{i=1}^{N}p_i
}
$$

This video-level probability is used for:

* threshold-based prediction
* F1
* precision
* recall
* fake recall
* accuracy
* confusion matrix

ROC-AUC is calculated using the continuous video-level probabilities.

Median and mode aggregation remain available in the existing infrastructure but are not primary Approach 4 conditions.

---

# 20. Fixed classification threshold

The classification threshold is:

$$
\boxed{0.5}
$$

This is inherited from the audited Approach 1 implementation.

The threshold remains fixed for:

* full FF++ models
* LOO FF++ models
* FF++ test evaluation
* Celeb-DF evaluation

No validation-set threshold optimization will be introduced.

There will be:

* no test-set threshold tuning
* no unseen-manipulation threshold tuning
* no Celeb-DF threshold tuning

This prevents the evaluation data from influencing the decision threshold.

---

# 21. Primary evaluation metrics

The following metrics are reported for every architecture, seed, manipulation, and training condition:

### Classification metrics

* Accuracy
* Precision
* Recall
* Fake-class recall
* F1-score
* ROC-AUC

### Error analysis

* Confusion matrix

The statistical unit for all primary evaluation is the video.

---

# 22. Primary generalization effects

For every manipulation \(M\), calculate:

### F1 effect

$$
\boxed{
\Delta F1_M =
F1_{Full,M}-F1_{LOO,M}
}
$$

### ROC-AUC effect

$$
\boxed{
\Delta AUC_M =
AUC_{Full,M}-AUC_{LOO,M}
}
$$

### Fake-recall effect

$$
\boxed{
\Delta Recall_{fake,M}
=
Recall_{fake,Full,M}
-
Recall_{fake,LOO,M}
}
$$

These three effects form the primary generalization analysis.

---

# 23. Interpretation of the direction of the effect

The analysis must not assume that the LOO model necessarily performs worse.

For any metric:

$$
\Delta > 0
$$

means the full-training model performs better.

$$
\Delta \approx 0
$$

means little measurable difference is observed.

$$
\Delta < 0
$$

means the LOO model performs better.

The implementation must preserve the signed effect.

A negative effect must not be converted to zero, removed, or described as a failed experiment.

Likewise, H3 is a directional hypothesis to be tested, not an expected result that the implementation is designed to confirm.

---

# 24. Bootstrap confidence intervals

For every full-versus-LOO comparison, bootstrap confidence intervals are calculated at the **video level**.

For a manipulation \(M\):

```text
Same test videos
        ↓
Full predictions
LOO predictions
        ↓
Paired video-level observations
        ↓
Resample videos with replacement
        ↓
Recalculate ΔF1 / ΔAUC / ΔRecall
        ↓
Bootstrap distribution
        ↓
95% CI
```

The exact bootstrap procedure should operate on video IDs, not frames.

The same video remains a single statistical observation regardless of how many frames were extracted from it.

---

# 25. Paired statistical testing

Where a valid pairing exists between full and LOO predictions on the same test videos, an appropriate paired statistical test may be applied.

The test is secondary to the effect size and confidence interval.

The analysis should report:

* observed effect
* confidence interval
* paired-test result where appropriate

The study should not add statistical tests solely to generate p-values for every available metric.

---

# 26. Seed aggregation

For each architecture and manipulation, the three seed-specific results are summarized as:

$$
\boxed{
mean \pm SD
}
$$

For example:

```text
Xception - NeuralTextures

Seed 42
Seed 123
Seed 2024

        ↓

Mean F1 ± SD
Mean AUC ± SD
Mean fake recall ± SD
```

The individual seed results remain available in the raw result files.

This allows the reader to distinguish:

* variability across training seeds
* uncertainty over the evaluated test videos

---

# 27. Per-manipulation results are primary

Results must be reported separately for:

* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

The primary results should not combine all four manipulations into a single generalizability number.

A macro-average may be provided as a secondary summary.

The macro-average must not replace the individual manipulation results.

---

# 28. Macro-average

If a macro-level summary is needed:

$$
Metric_{macro}
=
\frac{1}{4}
\sum_{M=1}^{4}Metric_M
$$

The four manipulation categories receive equal weight.

Macro-averaging is supplementary because it can conceal differences between manipulation techniques.

The report should therefore show the four individual effects first.

---

# 29. Main experimental matrix

The complete LOO experiment is:

| Held-out manipulation | Xception | EfficientNet-B0 | ResNet50 | Seeds         |
| --------------------- | -------: | --------------: | -------: | ------------- |
| Deepfakes             |        3 |               3 |        3 | 42, 123, 2024 |
| Face2Face             |        3 |               3 |        3 | 42, 123, 2024 |
| FaceSwap              |        3 |               3 |        3 | 42, 123, 2024 |
| NeuralTextures        |        3 |               3 |        3 | 42, 123, 2024 |
| **Total**             |   **12** |          **12** |   **12** | **36 runs**   |

The nine full-training Approach 1 models are reused where valid.

---

# 30. Full-versus-LOO result structure

For each manipulation:

```text
                    Same test condition
                         Original + M
                              │
               ┌──────────────┴──────────────┐
               ↓                             ↓
          Full model                    LOO model
          M was seen                    M was unseen
               │                             │
               └──────────────┬──────────────┘
                              ↓
                    Video-level predictions
                              ↓
                   F1 / AUC / Fake Recall
                              ↓
                    Full - LOO difference
                              ↓
                       Bootstrap 95% CI
```

This is the central analysis of Approach 4.

---

# 31. Cross-dataset experiment

The cross-dataset experiment is independent of the LOO training experiments.

The training distribution is the full FF++ distribution:

```text
Original
+ Deepfakes
+ Face2Face
+ FaceSwap
+ NeuralTextures
```

The valid Approach 1 full-training checkpoints are reused.

No new cross-dataset training is required unless checkpoint auditing identifies missing or invalid full-training checkpoints.

---

# 32. Cross-dataset evaluation workflow

The complete workflow is:

```text
Full FF++ training
        ↓
FF++ validation
        ↓
Lowest validation-loss checkpoint
        ↓
Freeze model
        ↓
FF++ test evaluation
        ↓
Celeb-DF v2 test evaluation
```

The Celeb-DF test data does not influence:

* checkpoint selection
* model selection
* threshold selection
* hyperparameter selection
* fine-tuning

---

# 33. Celeb-DF dataset protocol

The official Celeb-DF v2 testing list:

```text
List_of_testing_videos.txt
```

is used to define the test set.

The same preprocessing pipeline is applied to Celeb-DF.

Where the Celeb-DF directory structure confirms the intended mapping:

```text
Celeb-real
YouTube-real
    ↓
Real

Celeb-synthesis
    ↓
Fake
```

No assumption should be made if the local dataset structure differs from the expected official organization.

---

# 34. Cross-dataset threshold

The 0.5 threshold remains frozen.

The model does not learn a new Celeb-DF threshold.

The purpose is to evaluate how the FF++-trained detector behaves when transferred directly to Celeb-DF.

This makes the cross-dataset experiment distinct from domain adaptation.

---

# 35. Cross-dataset metrics

For every architecture and seed, record:

* FF++ accuracy
* Celeb-DF accuracy
* FF++ precision
* Celeb-DF precision
* FF++ recall
* Celeb-DF recall
* FF++ fake recall
* Celeb-DF fake recall
* FF++ F1
* Celeb-DF F1
* FF++ ROC-AUC
* Celeb-DF ROC-AUC
* FF++ confusion matrix
* Celeb-DF confusion matrix

Performance changes can be calculated as:

$$
\Delta F1_{dataset}
=
F1_{CelebDF}-F1_{FF++}
$$

and:

$$
\Delta AUC_{dataset}
=
AUC_{CelebDF}-AUC_{FF++}
$$

The direction must be preserved rather than assuming that cross-dataset performance decreases.

---

# 36. Interpretation of cross-dataset results

Cross-dataset performance differences should be described as changes under transfer from FF++ to Celeb-DF.

They should not automatically be attributed to a single cause.

The two datasets differ in aspects including:

* dataset composition
* identities and content
* manipulation-generation processes

Therefore the experiment measures **cross-dataset generalization**, not an isolated effect of one controlled domain variable.

---

# 37. Raw prediction storage

The implementation must retain video-level prediction data.

A minimum prediction record should contain:

```text
video_id
ground_truth
video_probability
video_prediction
architecture
seed
training_condition
held_out_manipulation
test_condition
dataset
```

For example:

```text
video_001
1
0.873
1
xception
42
LOO
NeuralTextures
unseen
FF++
```

This is required for reproducible statistical analysis.

It also prevents the need to rerun inference if additional video-level analysis is required.

---

# 38. Grad-CAM analysis

The existing Approach 3 Grad-CAM infrastructure is reused.

No new Grad-CAM implementation should be created unless the existing implementation cannot support the required LOO model.

The same:

* architecture
* target layer
* preprocessing
* class target
* Grad-CAM method
* checkpoint

must be retained when comparing seen and unseen conditions.

---

# 39. Grad-CAM comparison

For a given LOO model:

```text
LOO-Deepfakes
       ↓
Deepfakes excluded from training
       ↓
Compare:
    seen manipulation
    vs
    unseen Deepfakes
```

The same structure is applied to each held-out manipulation where feasible.

The purpose is to examine whether the spatial evidence associated with model predictions changes when the manipulation is unseen.

The analysis does not claim that Grad-CAM identifies the causal features used by the CNN.

---

# 40. Deterministic Grad-CAM sample selection

Samples are selected before visual inspection.

The procedure is:

```text
Eligible videos
      ↓
Sort video IDs
      ↓
Apply predefined eligibility criteria
      ↓
Select first N
      ↓
Generate Grad-CAM
```

The value of \(N\) should remain fixed across comparable conditions.

This prevents cherry-picking visually interesting examples after observing the heatmaps.

---

# 41. Grad-CAM quantitative analysis

Quantitative analysis uses the existing Approach 3 explainability metrics where applicable.

Potential measures already established for the project include:

* saliency overlap
* IoU-based spatial comparison
* heatmap correlation
* face-region saliency fraction

The exact metrics used must remain consistent with the finalized Approach 3 implementation.

No new metric should be introduced solely because it produces a more favorable result.

---

# 42. Grad-CAM qualitative analysis

A small deterministic set of examples is presented as visual panels.

The comparison should show:

```text
Input image
+
Grad-CAM
+
prediction
+
ground truth
```

for seen and unseen manipulation conditions.

The interpretation should remain descriptive.

For example:

> The Grad-CAM maps can be examined for changes in the spatial regions associated with the model's prediction under seen and unseen manipulation conditions.

The analysis should not claim that a heatmap proves the detector has learned manipulation-independent forensic features.

---

# 43. Research hypotheses

### H3

> **Excluding a manipulation category from training will reduce the detector's performance on that manipulation relative to performance when that manipulation is represented during training.**

This is a directional hypothesis.

The implementation must allow all three outcomes:

$$
\Delta F1 > 0
$$

Performance is higher when the manipulation is represented during training.

$$
\Delta F1 \approx 0
$$

Little measurable performance difference is observed.

$$
\Delta F1 < 0
$$

The LOO model performs better on the held-out manipulation.

The same principle applies to ROC-AUC and fake-class recall.

### H4

> **The magnitude of performance degradation caused by withholding a manipulation category will differ across manipulation techniques.**

H4 is evaluated by comparing the signed effects:

$$
\Delta F1_{DF},
\Delta F1_{F2F},
\Delta F1_{FS},
\Delta F1_{NT}
$$

and corresponding AUC and fake-recall effects.

The experiment does not assume that all manipulation techniques will exhibit the same effect.

---

# 44. What the experiment can and cannot establish

### It can evaluate

* whether performance changes when a manipulation is withheld
* how large that change is
* whether the change differs across manipulation categories
* how results vary across seeds
* uncertainty over the evaluated videos
* whether performance changes under cross-dataset evaluation
* whether prediction-associated spatial evidence changes between seen and unseen conditions

### It cannot independently establish

* that the CNN has learned manipulation-invariant forensic features
* that Grad-CAM regions are causal features
* that FF++ to Celeb-DF performance differences arise from a single isolated domain factor
* that the results generalize to all deepfake datasets
* that one architecture is universally superior

These limitations should remain explicit in the report.

---

# 45. Required result tables

## Table 1 - Full versus LOO performance

For each architecture and manipulation:

| Manipulation   | Architecture | Full F1 | LOO F1 | ΔF1 | Full AUC | LOO AUC | ΔAUC | ΔFake Recall |
| -------------- | ------------ | ------: | -----: | --: | -------: | ------: | ---: | -----------: |
| Deepfakes      | Xception     |         |        |     |          |         |      |              |
| Face2Face      | Xception     |         |        |     |          |         |      |              |
| FaceSwap       | Xception     |         |        |     |          |         |      |              |
| NeuralTextures | Xception     |         |        |     |          |         |      |              |

Values can be represented as mean ± SD across the three seeds, with the effect calculated using the corresponding seed-level predictions.

## Table 2 - Video bootstrap uncertainty

| Manipulation   | Architecture | ΔF1 | 95% CI | ΔAUC | 95% CI | ΔFake Recall | 95% CI |
| -------------- | ------------ | --: | ------ | ---: | ------ | -----------: | ------ |
| Deepfakes      | Xception     |     |        |      |        |              |        |
| Face2Face      | Xception     |     |        |      |        |              |        |
| FaceSwap       | Xception     |     |        |      |        |              |        |
| NeuralTextures | Xception     |     |        |      |        |              |        |

The bootstrap observations are videos, not seeds.

## Table 3 - Seed variability

| Manipulation   | Architecture | Seed 42 | Seed 123 | Seed 2024 | Mean | SD |
| -------------- | ------------ | ------: | -------: | --------: | ---: | -: |
| Deepfakes      | Xception     |         |          |           |      |    |
| Face2Face      | Xception     |         |          |           |      |    |
| FaceSwap       | Xception     |         |          |           |      |    |
| NeuralTextures | Xception     |         |          |           |      |    |

## Table 4 - Cross-dataset performance

| Architecture | Seed | FF++ F1 | Celeb-DF F1 | ΔF1 | FF++ AUC | Celeb-DF AUC | ΔAUC |
| ------------ | ---: | ------: | ----------: | --: | -------: | -----------: | ---: |
| Xception     |   42 |         |             |     |          |              |      |
| Xception     |  123 |         |             |     |          |              |      |
| Xception     | 2024 |         |             |     |          |              |      |

---

# 46. Required figures

The final analysis should include figures that directly answer the research questions.

### Figure 1 - Full versus LOO F1

A per-manipulation comparison showing:

```text
Deepfakes
Face2Face
FaceSwap
NeuralTextures
```

with full and LOO performance.

### Figure 2 - Generalization effect

Plot:

$$
\Delta F1
$$

for each manipulation, including uncertainty intervals.

The same type of figure can be generated for:

$$
\Delta AUC
$$

if useful.

### Figure 3 - Architecture comparison

Show how the LOO effect varies across:

* Xception
* EfficientNet-B0
* ResNet50

without producing an overall architecture ranking.

### Figure 4 - Cross-dataset performance

Compare FF++ and Celeb-DF performance for each architecture.

### Figure 5 - Grad-CAM panels

Show deterministic seen-versus-unseen examples.

The number of figures should be driven by the results rather than by a fixed requirement to produce a particular number.

---

# 47. Quality-control checks

Before accepting the results, the pipeline must automatically verify the following.

### Dataset checks

* Expected project subset is available.
* Video IDs are unique.
* Categories are correctly assigned.
* Expected category inventory is confirmed.
* Required frames/faces are available.

### Split checks

* No train/test video overlap.
* No train/validation overlap.
* No validation/test overlap.
* No relationship-connected component crosses splits.
* Held-out manipulation is absent from LOO training.
* Held-out manipulation is absent from LOO validation.
* Held-out manipulation is present in the intended unseen test condition.

### Training checks

* Correct architecture.
* Correct seed.
* ImageNet initialization.
* Correct hyperparameters.
* Correct LOO class weights.
* Validation loss used for checkpoint selection.
* Test performance not used for checkpoint selection.

### Evaluation checks

* Mean frame probability used.
* Video is the statistical unit.
* Threshold = 0.5.
* No test threshold tuning.
* Raw video probabilities saved.
* Full and LOO models evaluated on identical manipulation-specific test videos.

### Statistical checks

* Bootstrap operates over videos.
* Seeds are not treated as bootstrap observations.
* Seed mean ± SD is calculated separately.
* Signed effects are retained.
* Negative effects are not discarded.
* Confidence intervals correspond to the actual video-level comparison.

### Cross-dataset checks

* Full FF++ model only.
* Frozen checkpoint.
* Official Celeb-DF testing list.
* No fine-tuning.
* No threshold tuning.
* Same preprocessing.
* Raw video-level probabilities saved.

---

# 48. Implementation order

The implementation should follow this order.

## Phase 1 - Repository and data audit

1. Inspect the actual `data/splits/splits.json`.
2. Verify all train/validation/test video IDs.
3. Calculate category counts per split.
4. Reconstruct or inspect connected-component assignments.
5. Verify component-level leakage protection.
6. Inspect `data/checkpoints/`.
7. Identify all nine Approach 1 checkpoints.
8. Verify architecture and seed for every checkpoint.
9. Verify checkpoint metadata.
10. Verify the selected epoch and validation loss.
11. Verify that checkpoints correspond to the full FF++ training distribution.
12. Inspect existing `data/output/` evaluation artifacts.
13. Confirm the existing video-level aggregation.
14. Confirm the 0.5 threshold.
15. Determine which Approach 1 checkpoints are valid for reuse.

**No Approach 4 training begins until this audit passes.**

---

## Phase 2 - LOO manifest generation

For each manipulation \(M\):

1. Start from the audited Approach 1 split.
2. Filter out \(M\) from training.
3. Filter out \(M\) from validation.
4. Keep Original and the remaining three manipulations.
5. Construct the test condition containing Original + \(M\).
6. Verify video IDs.
7. Verify counts.
8. Verify no overlap.
9. Verify that the held-out manipulation is completely absent from training and validation.
10. Save the resulting manifest.

Four validated LOO manifests should exist before training starts.

---

## Phase 3 - Training pipeline validation

Before running all 36 experiments:

1. Select one pilot condition.
2. Use:

   * Xception
   * seed 42
   * NeuralTextures held out
3. Run the complete training pipeline.
4. Verify training distribution.
5. Verify validation distribution.
6. Verify NeuralTextures exclusion.
7. Verify class-weight calculation.
8. Verify checkpoint selection.
9. Verify video-level aggregation.
10. Verify threshold = 0.5.
11. Verify prediction storage.
12. Verify evaluation metrics.

Only after this pilot is correct should the full LOO matrix be executed.

---

# 49. Full LOO execution

After the pilot passes:

```text
4 held-out manipulations
        ×
3 architectures
        ×
3 seeds
        =
36 training runs
```

Each run must produce:

* configuration metadata
* training history
* best checkpoint
* validation metrics
* test predictions
* video-level probabilities
* video-level predictions
* evaluation metrics
* confusion matrix

The experiment identifier should encode at least:

```text
architecture
seed
held_out_manipulation
```

---

# 50. Statistical aggregation pipeline

After all LOO models finish:

1. Load full-model predictions.
2. Load LOO predictions.
3. Match predictions by video ID.
4. Verify identical test-video sets.
5. Calculate F1 for both conditions.
6. Calculate ROC-AUC for both conditions.
7. Calculate fake recall for both conditions.
8. Calculate signed full-minus-LOO effects.
9. Bootstrap paired video-level observations.
10. Generate 95% confidence intervals.
11. Apply appropriate paired tests where valid.
12. Aggregate the three seeds using mean ± SD.
13. Generate per-manipulation summaries.
14. Generate optional macro summaries.

The aggregation stage must never treat frames as independent observations.

---

# 51. Cross-dataset execution

After the full FF++ checkpoint audit:

1. Identify valid full-training checkpoints.
2. Obtain the official Celeb-DF v2 test-list information.
3. Verify local Celeb-DF directory structure.
4. Map labels based on the confirmed dataset structure.
5. Apply the existing preprocessing pipeline.
6. Run inference using frozen FF++ checkpoints.
7. Use mean frame probability.
8. Apply the fixed 0.5 threshold.
9. Save raw video-level probabilities.
10. Calculate metrics.
11. Compare FF++ and Celeb-DF performance.
12. Aggregate seed results using mean ± SD.

No model adaptation occurs during this phase.

---

# 52. Grad-CAM execution

After the performance experiments:

1. Select the relevant LOO checkpoints.
2. Define eligibility criteria before visual inspection.
3. Sort eligible video IDs.
4. Select the first \(N\).
5. Generate Grad-CAM maps using the existing implementation.
6. Evaluate the selected seen and unseen conditions.
7. Calculate the established quantitative explainability measures.
8. Generate qualitative panels.
9. Store the selected video IDs.
10. Store the corresponding predictions and Grad-CAM outputs.

This makes the explainability experiment reproducible.

---

# 53. Final analysis structure

The final Approach 4 analysis should answer four questions.

### Question 1

**Does excluding a manipulation from training affect performance on that manipulation?**

Answered using:

$$
\Delta F1,\ \Delta AUC,\ \Delta Recall_{fake}
$$

with video-level bootstrap confidence intervals.

### Question 2

**Does the effect differ across manipulation techniques?**

Answered using the four manipulation-specific effects.

### Question 3

**Does a detector trained on FF++ generalize to Celeb-DF?**

Answered using the independent cross-dataset experiment.

### Question 4

**Does the prediction-associated spatial evidence change between seen and unseen manipulation conditions?**

Answered using the secondary Grad-CAM analysis.

---

# 54. Final locked experimental design

The complete Approach 4 design is:

```text
                    APPROACH 1
                Full FF++ models
              3 architectures × 3 seeds
                         │
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
           DF test     F2F test     FS test ... NT test
             │           │           │
             └───────────┴───────────┘
                         │
                         ↓
                  Full baseline
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
       LOO experiments          Same test sets
             │
     ┌───────┼────────┬────────┐
     ↓       ↓        ↓        ↓
    LOO-DF  LOO-F2F  LOO-FS  LOO-NT
     │       │        │        │
     └───────┴────────┴────────┘
                 │
                 ↓
        Full vs LOO comparison
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
      ΔF1      ΔAUC    ΔFake Recall
       │         │         │
       └─────────┼─────────┘
                 ↓
       Paired video predictions
                 ↓
       Bootstrap over videos
                 ↓
             95% CI
```

The seed analysis runs alongside this:

```text
Seed 42
Seed 123
Seed 2024
    ↓
mean ± SD
    ↓
training stochasticity
```

These two uncertainty analyses remain separate.

The cross-dataset experiment is independent:

```text
Full FF++ model
       ↓
Validation-selected checkpoint
       ↓
Freeze
       ↓
FF++ test
       ↓
Celeb-DF v2
       ↓
Cross-dataset performance
```

The explainability experiment is also independent:

```text
LOO checkpoint
      ↓
Deterministically selected videos
      ↓
Seen vs unseen manipulation
      ↓
Existing Grad-CAM implementation
      ↓
Quantitative + qualitative analysis
```

## Locked parameters

| Component                        | Final decision                                 |
| -------------------------------- | ---------------------------------------------- |
| Dataset                          | 650-video FF++ C23 project subset              |
| Manipulations                    | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| Primary generalization design    | Full vs LOO                                    |
| LOO conditions                   | 4                                              |
| Architectures                    | Xception, EfficientNet-B0, ResNet50            |
| Seeds                            | 42, 123, 2024                                  |
| New LOO runs                     | 36                                             |
| Full-model baseline              | Existing valid Approach 1 checkpoints          |
| Split                            | Existing leakage-safe split after audit        |
| LOO training                     | Original + 3 seen manipulations                |
| LOO test                         | Original + 1 unseen manipulation               |
| Primary statistical unit         | Video                                          |
| Frame aggregation                | Mean probability                               |
| Threshold                        | Fixed 0.5                                      |
| Checkpoint selection             | Lowest validation loss                         |
| Test-informed seed selection     | Not allowed                                    |
| Primary metrics                  | F1, ROC-AUC, fake recall                       |
| Additional metrics               | Accuracy, precision, recall, confusion matrix  |
| Seed summary                     | Mean ± SD                                      |
| Uncertainty                      | Video-level bootstrap 95% CI                   |
| Paired testing                   | Where statistically appropriate                |
| Primary effect                   | Full - LOO                                     |
| Macro-average                    | Supplementary                                  |
| Single generalizability score    | Not used                                       |
| Cross-dataset                    | Full FF++ → Celeb-DF v2                        |
| Celeb-DF fine-tuning             | Not allowed                                    |
| Celeb-DF threshold tuning        | Not allowed                                    |
| Grad-CAM                         | Existing Approach 3 implementation             |
| Grad-CAM selection               | Deterministic first-N after sorting            |
| Grad-CAM role                    | Secondary evidence                             |
| Primary hypothesis               | H3                                             |
| Manipulation-specific hypothesis | H4                                             |
