# Approach 2 plan: robustness under image transformations

## 1. Purpose

Approach 2 evaluates the robustness of the CNN deepfake detectors
trained in Approach 1 when their input face crops are subjected to
controlled image-level corruptions.

The experiment is intentionally restricted to **input-corruption
robustness**. Transformations are applied after face extraction and
before the model-specific input resize. RetinaFace detection, face
tracking, frame selection, dataset splitting, model weights, and
training procedures are not changed.

The central experimental question is:

> How does the performance, confidence, prediction stability, and
> per-manipulation behavior of the trained CNN detectors change as
> controlled image corruption severity increases?

Approach 2 is an evaluation-only experiment. No checkpoint is retrained,
fine-tuned, or modified.

The design follows the cumulative structure of the project:

``` text
Approach 1: CNN baseline
        |
        v
Approach 2: robustness under controlled image transformations
        |
        v
Approach 3: explainability
        |
        v
Approach 4: generalizability
```

The clean Approach 1 test performance is the reference condition for
every robustness experiment.

------------------------------------------------------------------------

## 2. Relationship to Approach 1

Approach 2 must remain directly comparable with the completed Approach 1
baseline.

The following components are inherited from Approach 1 and must not be
changed:

-   FaceForensics++ C23 dataset subset.
-   Existing leakage-safe train/validation/test split.
-   Existing test manifest.
-   Existing extracted face crops.
-   Existing frame-selection procedure.
-   Existing face-detection and tracking pipeline.
-   Existing binary Real/Fake target.
-   Existing Xception, EfficientNet-B0, and ResNet50 architectures.
-   Existing trained checkpoints.
-   Existing model-specific input dimensions.
-   Existing ImageNet normalization.
-   Existing video-level evaluation protocol.
-   Existing primary mean-probability video aggregation.
-   Existing secondary median and mode aggregation.
-   Existing metric definitions.

Approach 2 adds a transformation layer to the inference path. It does
not replace the baseline pipeline.

------------------------------------------------------------------------

## 3. Research scope

### 3.1 What is being tested

The primary object of study is:

**CNN sensitivity to controlled visual corruption of already extracted
face crops.**

The experiment therefore measures robustness of the learned detector to
input distribution changes.

### 3.2 What is not being tested

Approach 2 does not evaluate:

-   RetinaFace detection robustness.
-   Face tracking robustness.
-   Failure of face extraction.
-   Failure of frame selection.
-   Failure of dataset splitting.
-   Robustness to unseen manipulation techniques.
-   Cross-dataset generalization.
-   Retraining with corrupted samples.
-   Adversarial robustness.
-   Model architecture changes.
-   Training-time robustness augmentation.

These belong to other parts of the project or are outside the current
research scope.

------------------------------------------------------------------------

## 4. Models and checkpoints

All existing Approach 1 checkpoints are evaluated.

  Architecture                    Seeds   Number of checkpoints
  ----------------- ------------------- -----------------------
  Xception                42, 123, 2024                       3
  EfficientNet-B0         42, 123, 2024                       3
  ResNet50                42, 123, 2024                       3
  **Total**           **3 seeds/model**                   **9**

Each checkpoint is treated as an independent experimental run.

Results are retained at two levels:

1.  **Checkpoint level** - preserves exact reproducibility and exposes
    seed variation.
2.  **Architecture level** - aggregates the three seeds of each
    architecture using mean and standard deviation.

No new model training is performed.

------------------------------------------------------------------------

## 5. Dataset and evaluation population

Approach 2 uses the existing **test manifest** from Approach 1.

The experiment must not regenerate the train/validation/test split.

The test population contains the same video-level identities and
manipulation categories used by Approach 1:

-   Original, Real
-   DeepFake, Fake
-   Face2Face, Fake
-   FaceSwap, Fake
-   NeuralTextures, Fake

The CNN remains a binary classifier:

``` text
0 = Real
1 = Fake
```

The manipulation category is metadata used for stratified analysis. It
is not a model target.

Every usable test frame is evaluated quantitatively.

For visual examples only, approximately every 50th **usable face crop**
is considered for candidate sampling.

------------------------------------------------------------------------

## 6. Core experimental principle

The corruption must be applied to the saved face crop before the
existing model-specific resize.

The complete inference path becomes:

``` text
saved face crop
      |
      v
robustness transformation
      |
      v
model-specific resize
      |
      v
tensor conversion
      |
      v
ImageNet normalization
      |
      v
CNN checkpoint
      |
      v
sigmoid fake probability
      |
      v
frame prediction
      |
      v
video aggregation
      |
      v
metrics and robustness analysis
```

For the clean reference:

``` text
saved face crop
      |
      v
model-specific resize
      |
      v
tensor conversion
      |
      v
ImageNet normalization
      |
      v
CNN checkpoint
```

This ordering is essential.

The transformation must not be applied after the final model input
resize because doing so would make the corruption severity dependent on
the architecture's input resolution.

------------------------------------------------------------------------

## 7. Transformation categories

Approach 2 contains seven transformation types.

### 7.1 Core transformations

These are enabled by default:

1.  JPEG compression.
2.  Lower-resolution resizing.
3.  Brightness changes.

### 7.2 Optional transformations

These are independently configurable and disabled by default:

4.  Gaussian noise.
5.  Gaussian blur.
6.  Contrast changes.
7.  Controlled centered cropping.

The optional transformations must not automatically become active merely
because the robustness evaluator is executed.

------------------------------------------------------------------------

## 8. Independent transformation experiments

Each transformation is evaluated independently from the clean image.

For example:

``` text
Clean -> JPEG severity 1
Clean -> JPEG severity 2
Clean -> JPEG severity 3
```

and separately:

``` text
Clean -> Resize severity 1
Clean -> Resize severity 2
Clean -> Resize severity 3
```

The default experiment must not perform:

``` text
JPEG -> Resize
JPEG -> Brightness
Resize -> Brightness
JPEG -> Resize -> Brightness
```

Transformation combinations may be added later as a separate
experimental mode, but they are not part of Approach 2's primary design.

This separation allows performance degradation to be attributed to a
specific transformation.

------------------------------------------------------------------------

## 9. Severity design

Each transformation uses:

-   Clean reference, severity 0.
-   Three corruption severity levels.

Therefore, the default structure for each one-dimensional transformation
is:

``` text
severity 0 = clean
severity 1 = mild
severity 2 = moderate
severity 3 = strong
```

The exact numerical parameters must be fixed before the full experiment
is executed.

They must be:

-   deterministic,
-   documented,
-   reproducible,
-   visually meaningful,
-   strong enough to produce measurable distribution shifts,
-   not so extreme that the transformed images become unrelated to
    realistic image degradation.

The same severity definitions must be used for all nine checkpoints.

A transformation parameter must never be selected separately for each
model.

------------------------------------------------------------------------

## 10. JPEG compression

### Objective

Measure sensitivity to JPEG compression artifacts.

### Procedure

For each clean face crop:

``` text
face crop
   |
   v
JPEG encode at selected quality
   |
   v
JPEG decode
   |
   v
transformed face crop
```

Lower JPEG quality corresponds to greater compression severity.

### Conditions

``` text
Clean
JPEG severity 1
JPEG severity 2
JPEG severity 3
```

The numerical JPEG quality values are configuration parameters and must
be recorded in the experiment configuration.

### Requirements

-   Use the same quality values for all models.
-   Use the same transformed image for all nine checkpoints.
-   Do not change the final CNN input size.
-   Store the quality value in prediction metadata.
-   Retain clean predictions as the reference.

------------------------------------------------------------------------

## 11. Lower-resolution resizing

### Objective

Measure sensitivity to loss of spatial resolution.

### Procedure

The transformation operates on the original saved face crop:

``` text
original face crop
       |
       v
downsample to severity-specific resolution
       |
       v
upsample to original crop dimensions
       |
       v
model-specific resize
       |
       v
CNN
```

The final model input dimensions remain unchanged.

For example, Xception continues to receive its existing input size,
while EfficientNet-B0 and ResNet50 continue to receive their existing
input size.

### Why upsample?

The purpose is to simulate information loss while preserving the same
external image dimensions expected by the rest of the preprocessing
pipeline.

Changing the CNN input dimensions would confound:

-   robustness to resolution loss,
-   model architecture,
-   model input preprocessing.

### Conditions

``` text
Clean
Resize severity 1
Resize severity 2
Resize severity 3
```

The resize scale parameters must be fixed in configuration.

------------------------------------------------------------------------

## 12. Brightness changes

Brightness is evaluated in both directions.

### Darkening

``` text
Clean
Dark severity 1
Dark severity 2
Dark severity 3
```

### Brightening

``` text
Clean
Bright severity 1
Bright severity 2
Bright severity 3
```

The two directions are retained separately in raw results.

A combined brightness summary may be produced for high-level analysis,
but darkening and brightening must not be merged in a way that hides
directional differences.

The transformation must be deterministic.

------------------------------------------------------------------------

## 13. Gaussian noise

Gaussian noise is optional and disabled by default.

### Procedure

A controlled Gaussian perturbation is applied to the image:

``` text
clean image + Gaussian noise
```

Three sigma levels are used.

### Reproducibility

A fixed seed must be used.

The same source frame and severity must produce the same noise
realization regardless of which CNN checkpoint receives the image.

This prevents the following unwanted situation:

``` text
Xception -> noise realization A
EfficientNet -> noise realization B
ResNet -> noise realization C
```

Instead:

``` text
same source frame + same severity
             |
             v
       same corrupted image
             |
       +-----+-----+
       |     |     |
       v     v     v
       X     E     R
```

------------------------------------------------------------------------

## 14. Gaussian blur

Gaussian blur is optional and disabled by default.

Three severity levels are used.

The configuration should explicitly store the blur parameters.

The preferred implementation exposes both:

-   kernel size,
-   sigma.

Severity must increase monotonically.

The implementation must ensure that kernel sizes satisfy the
requirements of the selected image-processing library.

------------------------------------------------------------------------

## 15. Contrast changes

Contrast changes are optional and disabled by default.

Three severity levels are used.

The clean image is the reference.

The experiment should define contrast factors relative to the clean
condition and preserve the same factor definitions across all models.

Raw results must retain the factor used for each image.

------------------------------------------------------------------------

## 16. Controlled cropping

Controlled cropping is optional and disabled by default.

Only centered cropping is used.

The procedure is:

``` text
original face crop
       |
       v
center crop
       |
       v
resize back to original crop dimensions
       |
       v
model-specific resize
```

Three crop severities correspond to increasing loss of peripheral image
information.

Random cropping is deliberately excluded because it introduces an
additional stochastic variable and makes the experiment less controlled.

------------------------------------------------------------------------

## 17. Transformation configuration

Robustness settings should be centralized.

A configuration structure should follow this conceptual form:

``` yaml
robustness:
  jpeg:
    enabled: true
    qualities: [...]

  resize:
    enabled: true
    scales: [...]

  brightness:
    enabled: true
    darker: [...]
    brighter: [...]

  gaussian_noise:
    enabled: false
    sigmas: [...]
    seed: 42

  gaussian_blur:
    enabled: false
    kernel_sizes: [...]
    sigmas: [...]

  contrast:
    enabled: false
    factors: [...]

  crop:
    enabled: false
    ratios: [...]
```

The exact numerical values must be frozen before the full experiment.

The configuration must be copied into each experiment's output
directory.

This prevents a result from becoming ambiguous if the global
configuration changes later.

------------------------------------------------------------------------

## 18. Core transformation enablement

The three core transformations are enabled by default because they are
explicitly reserved as the primary robustness tests:

``` text
JPEG compression       = enabled
Lower-resolution       = enabled
Brightness             = enabled
```

The four optional transformations are:

``` text
Gaussian noise         = disabled
Gaussian blur          = disabled
Contrast               = disabled
Centered crop          = disabled
```

The evaluator should still support explicit configuration of every
transformation so that experiments can be reproduced without code
modification.

------------------------------------------------------------------------

## 19. Clean baseline evaluation

Clean inference is performed once per checkpoint.

For each checkpoint:

1.  Load the test manifest.
2.  Load the checkpoint.
3.  Run clean frame-level inference.
4.  Store frame-level fake probabilities.
5.  Aggregate frame predictions to video predictions.
6.  Calculate clean metrics.
7.  Cache the clean results.
8.  Reuse the cached results for every transformation comparison.

The clean baseline must not be recomputed independently for every
transformation.

This provides a common reference:

``` text
clean result
    |
    +--> JPEG severity 1
    +--> JPEG severity 2
    +--> JPEG severity 3
    |
    +--> Resize severity 1
    +--> Resize severity 2
    +--> Resize severity 3
    |
    +--> Brightness ...
```

------------------------------------------------------------------------

## 20. Model-specific preprocessing

The transformation occurs before model-specific resizing.

The final resize remains determined by the checkpoint architecture.

Conceptually:

``` python
image = load_face_crop(path)
image = apply_robustness_transform(image, condition)
image = resize_for_model(image, model_name)
image = normalize(image)
prediction = model(image)
```

The robustness transformation must not directly modify the model's
configured input dimensions.

------------------------------------------------------------------------

## 21. Dynamic transformations

Transformed images should be generated dynamically in memory rather than
stored as a complete transformed dataset.

Advantages:

-   avoids duplicating the face-crop dataset,
-   avoids large additional disk usage,
-   makes transformation configuration easier,
-   makes optional transformations independently controllable,
-   keeps clean data unchanged,
-   ensures transformations are generated from the same clean source.

A small set of visual examples may be saved separately for inspection
and reporting.

------------------------------------------------------------------------

## 22. Shared transformed inputs across models

For a given:

``` text
source frame
+
transformation
+
severity
```

the resulting corrupted image should be identical for every checkpoint.

The model-specific resize occurs only after the shared corruption step.

Therefore:

``` text
clean crop
    |
    +--> JPEG Qx
            |
            +--> Xception resize
            +--> EfficientNet resize
            +--> ResNet resize
```

This is required for a fair architecture comparison.

------------------------------------------------------------------------

## 23. Frame-level predictions

Every quantitative robustness condition must retain frame-level
predictions.

A prediction record should contain at least:

``` text
video_id
frame_path
category
label
transformation
severity
direction
prob_fake
pred_fake
```

Additional metadata may include:

``` text
model
seed
checkpoint
parameter_value
```

For clean:

``` text
transformation = clean
severity = 0
direction = none
```

Frame-level fake probabilities are retained because they support later
analyses without rerunning inference.

------------------------------------------------------------------------

## 24. Video-level aggregation

The existing Approach 1 hierarchy is preserved.

### Primary aggregation

Mean fake probability across frames:

``` text
P_video = mean(P_frame)
```

The binary video prediction is obtained using the existing thresholding
procedure.

### Secondary aggregation

Also preserve:

-   median aggregation,
-   mode aggregation.

Mean remains the primary method so that clean and robustness results
remain directly comparable with Approach 1.

Median and mode are secondary diagnostics.

------------------------------------------------------------------------

## 25. Primary metrics

For every:

``` text
model
seed
transformation
severity
```

calculate:

-   Accuracy.
-   Precision.
-   Recall.
-   F1.
-   ROC-AUC.
-   Confusion matrix.

Results must be available at:

1.  overall test-set level,
2.  per-manipulation level.

The primary robustness analysis focuses on F1 and ROC-AUC degradation.

------------------------------------------------------------------------

## 26. Performance degradation

For each transformed condition, calculate the difference from the
corresponding clean result.

For F1:

``` text
Delta_F1 =
F1_transformed - F1_clean
```

For ROC-AUC:

``` text
Delta_ROC_AUC =
ROC_AUC_transformed - ROC_AUC_clean
```

Similarly calculate changes in:

-   Accuracy.
-   Precision.
-   Recall.

A negative delta indicates degradation.

Both absolute transformed performance and degradation relative to clean
must be reported.

Example:

``` text
Condition       F1       Delta F1
Clean           0.91     0.00
JPEG severity1  0.89    -0.02
JPEG severity2  0.84    -0.07
JPEG severity3  0.76    -0.15
```

------------------------------------------------------------------------

## 27. Severity-response analysis

For each transformation, plot performance as severity increases.

Example:

``` text
Severity 0 -> Severity 1 -> Severity 2 -> Severity 3
```

Generate curves for at least:

-   F1,
-   ROC-AUC.

Secondary curves may include:

-   Accuracy,
-   Precision,
-   Recall.

A corresponding degradation curve should also be available.

The clean condition must be shown as severity 0.

------------------------------------------------------------------------

## 28. Seed analysis

Each architecture has three seeds.

For every transformation and severity:

``` text
seed 42
seed 123
seed 2024
```

are retained separately.

Then calculate:

``` text
mean
standard deviation
```

across seeds.

The report should distinguish:

-   performance degradation caused by the transformation,
-   variation caused by model initialization/training seed.

This prevents a single checkpoint from being treated as representative
of an architecture.

------------------------------------------------------------------------

## 29. Architecture comparison

Compare:

-   Xception,
-   EfficientNet-B0,
-   ResNet50.

For each architecture, report mean ± SD across the three seeds.

Useful comparisons include:

``` text
Architecture
    |
    +--> Clean F1
    +--> Transformed F1
    +--> Delta F1
    +--> Clean ROC-AUC
    +--> Transformed ROC-AUC
    +--> Delta ROC-AUC
```

The architecture ranking is descriptive.

A model should not be called globally "most robust" based on one
transformation or one metric.

------------------------------------------------------------------------

## 30. Per-manipulation robustness

For fake samples, calculate robustness separately for:

-   DeepFake,
-   Face2Face,
-   FaceSwap,
-   NeuralTextures.

The Real category should also be retained for false-positive analysis.

For each manipulation:

``` text
clean performance
        vs
transformed performance
        vs
severity
```

This can reveal cases where a detector is stable for one manipulation
type but sensitive to another.

The analysis remains binary classification.

------------------------------------------------------------------------

## 31. False-positive and false-negative analysis

Calculate transformation-induced changes in:

### False-positive rate

Real videos incorrectly classified as Fake.

### False-negative rate

Fake videos incorrectly classified as Real.

Report changes relative to clean.

This is useful because an unchanged F1 score can hide a meaningful shift
in error type.

For example, a transformation may reduce false positives while
increasing false negatives.

------------------------------------------------------------------------

## 32. Confusion-matrix analysis

Generate confusion matrices for clean and transformed conditions.

At minimum, compare:

``` text
Clean
JPEG severity 1
JPEG severity 2
JPEG severity 3
```

and equivalent conditions for the other enabled transformations.

Confusion matrices should also be available for architecture-level
summaries where appropriate.

------------------------------------------------------------------------

## 33. Prediction-flip analysis

For every video, compare its clean prediction with its transformed
prediction.

Define:

``` text
flip = transformed_prediction != clean_prediction
```

Then calculate:

``` text
video_flip_rate =
number of videos whose prediction changes
/
number of evaluated videos
```

Report flip rate by:

-   model,
-   seed,
-   transformation,
-   severity,
-   manipulation category.

A low flip rate indicates stable binary decisions, but it must not be
interpreted as complete robustness because confidence can change without
crossing the decision threshold.

------------------------------------------------------------------------

## 34. Confidence analysis

The continuous fake probability must be retained.

For every video and condition, calculate statistics such as:

-   mean fake probability,
-   median fake probability,
-   standard deviation,
-   probability shift from clean.

Define:

``` text
Delta_probability =
P_transformed - P_clean
```

Analyze probability shifts by:

-   transformation,
-   severity,
-   architecture,
-   seed,
-   manipulation category,
-   clean correctness.

This allows us to detect cases where:

``` text
clean prediction = Fake
transformed prediction = Fake
```

but the fake probability falls substantially.

Such a case is prediction-stable but confidence-sensitive.

------------------------------------------------------------------------

## 35. Frame-level prediction flips

In addition to video-level flips, calculate the fraction of frames whose
binary prediction changes relative to the corresponding clean frame.

This provides a finer-grained diagnostic:

``` text
frame_flip_rate =
changed frame predictions
/
corresponding evaluated frames
```

Video-level flip rate remains the primary stability measure.

Frame-level flip rate is supplementary because frames from the same
video are correlated.

------------------------------------------------------------------------

## 36. Error-state transition analysis

For every video, classify the clean/transformed pair into four states:

  Clean       Transformed   Category
  ----------- ------------- -----------------------------------
  Correct     Correct       Stable correct
  Correct     Incorrect     Robustness failure
  Incorrect   Correct       Transformation-induced correction
  Incorrect   Incorrect     Persistent error

Calculate the proportion of videos in each category.

The key quantity for robustness failure analysis is:

``` text
clean-correct -> transformed-incorrect
```

This separates genuine transformation-induced failures from
transformations that happen to correct an existing baseline error.

------------------------------------------------------------------------

## 37. Paired statistical analysis

The statistical unit is the **video**.

Frames must not be treated as independent videos because multiple frames
originate from the same video.

For clean versus transformed comparisons, use paired video-level
observations.

For each transformation and severity, calculate bootstrap confidence
intervals for quantities such as:

-   Delta F1,
-   Delta ROC-AUC,
-   prediction-flip rate,
-   confidence shift.

The bootstrap resamples videos, not individual frames.

Seed-level mean ± SD remains a separate analysis.

The statistical analysis supports interpretation but does not replace
the primary performance measurements.

------------------------------------------------------------------------

## 38. Robustness ranking

A secondary descriptive ranking may be produced.

Possible ranking dimensions include:

-   average Delta F1,
-   average Delta ROC-AUC,
-   severity-response area,
-   prediction-flip rate.

However, the project must not claim that a single scalar value
represents universal robustness.

Different transformations have different parameter scales and affect
images through different mechanisms.

Therefore:

``` text
transformation-specific results
        >
single global robustness score
```

A summary ranking can be shown as a descriptive convenience only.

------------------------------------------------------------------------

## 39. Area under the performance-severity curve

For transformations with ordered severity levels, calculate an
area-under-curve style summary where appropriate.

The curve may be:

``` text
performance vs normalized severity
```

or:

``` text
degradation vs normalized severity
```

The exact implementation must be documented so that curves with
different parameter ranges are comparable only when normalization is
justified.

This metric is secondary to the actual severity curves.

------------------------------------------------------------------------

## 40. Relationship between clean performance and robustness

Analyze whether a checkpoint with stronger clean performance necessarily
has lower robustness degradation.

For each checkpoint, compare:

``` text
clean F1
        vs
average Delta F1
```

and similarly for ROC-AUC.

This helps distinguish:

-   high baseline performance,
-   low degradation,
-   high absolute transformed performance.

These are related but not equivalent properties.

------------------------------------------------------------------------

## 41. Visual quality verification

Quantitative corruption experiments should be accompanied by
automatically generated candidate visual examples.

For visual sampling:

-   operate on usable face crops,
-   approximately select every 50th usable face crop,
-   generate examples for each transformation/severity,
-   retain representative examples for the final report.

The purpose is not to perform visual classification.

The purpose is to verify that:

1.  severity increases in the intended direction,
2.  the transformation is actually applied,
3.  no unexpected artifacts are introduced by implementation,
4.  the resulting corruption remains interpretable.

------------------------------------------------------------------------

## 42. Visual example generation

For each selected source image, produce a sequence such as:

``` text
Clean | Severity 1 | Severity 2 | Severity 3
```

For brightness:

``` text
Clean | Dark 1 | Dark 2 | Dark 3
Clean | Bright 1 | Bright 2 | Bright 3
```

The automatic generator may create many candidate examples.

Only representative examples should be included in the final report.

The full candidate set does not need to be committed to Git if it is
large.

------------------------------------------------------------------------

## 43. Reproducibility requirements

Every robustness experiment must record:

-   Git commit/branch.
-   Model architecture.
-   Checkpoint path or identifier.
-   Seed.
-   Test manifest path.
-   Transformation name.
-   Severity level.
-   Exact transformation parameter.
-   Random seed where applicable.
-   Model input size.
-   Evaluation aggregation method.
-   Evaluation threshold.
-   Timestamp.
-   Configuration snapshot.

For deterministic transformations, repeated execution with the same
configuration should produce the same transformed image.

For Gaussian noise, the fixed seed must ensure reproducible corruption.

------------------------------------------------------------------------

## 44. Suggested code organization

The implementation should extend the existing Approach 1 code rather
than creating a separate framework.

Recommended structure:

``` text
src/
├── config.py
├── dataset.py
├── evaluate.py
├── robustness.py
├── robustness_analysis.py
├── run_robustness.py
└── ...
```

### `src/robustness.py`

Responsible for:

-   transformation implementations,
-   transformation parameter validation,
-   deterministic random state handling,
-   transformation dispatch,
-   severity metadata.

It should not calculate model metrics.

### `src/run_robustness.py`

Responsible for:

-   loading configuration,
-   discovering checkpoints,
-   loading the test manifest,
-   running clean inference,
-   running transformed inference,
-   saving raw predictions,
-   coordinating analysis.

It should not contain large transformation implementations.

### `src/robustness_analysis.py`

Responsible for:

-   metric comparison,
-   degradation calculations,
-   seed aggregation,
-   architecture aggregation,
-   manipulation-level analysis,
-   confidence analysis,
-   prediction-flip analysis,
-   error-state transitions,
-   bootstrap confidence intervals,
-   robustness summaries.

### Existing `src/evaluate.py`

Reuse existing components where practical for:

-   checkpoint loading,
-   inference,
-   frame prediction,
-   video aggregation,
-   metric calculation.

Do not duplicate functionality unnecessarily.

------------------------------------------------------------------------

## 45. Transformation interface

The transformation layer should expose a consistent interface.

Conceptually:

``` python
transformed = apply_robustness_transform(
    image,
    transformation_name,
    severity,
    parameters,
    seed,
)
```

The returned image must have the same basic image representation
expected by the downstream preprocessing code.

The function should not:

-   load model checkpoints,
-   perform normalization,
-   resize to CNN input dimensions,
-   calculate metrics,
-   save predictions.

This separation keeps transformation behavior testable.

------------------------------------------------------------------------

## 46. Validation requirements

Before running all nine checkpoints, each transformation should be
unit-tested.

Tests should verify:

### JPEG

-   output is readable,
-   dimensions remain correct,
-   quality levels are ordered by intended severity.

### Resize

-   output dimensions return to the original crop dimensions,
-   downsample/upsample sequence is correct,
-   severity levels are ordered.

### Brightness

-   dark levels reduce brightness,
-   bright levels increase brightness,
-   clean remains unchanged.

### Gaussian noise

-   fixed seed produces identical output,
-   changing seed changes the noise,
-   severity changes noise magnitude.

### Gaussian blur

-   output dimensions remain unchanged,
-   blur increases with severity.

### Contrast

-   output dimensions remain unchanged,
-   factors produce the expected direction of contrast change.

### Crop

-   crop is centered,
-   output dimensions are restored,
-   crop severity increases information removal.

------------------------------------------------------------------------

## 47. Pipeline validation before full execution

A single checkpoint should be used for an initial end-to-end validation.

Recommended sequence:

``` text
one checkpoint
    |
    +--> clean
    +--> JPEG 3 levels
    +--> resize 3 levels
    +--> brightness dark 3 levels
    +--> brightness bright 3 levels
```

Verify:

-   no preprocessing errors,
-   no corrupted outputs,
-   expected image dimensions,
-   correct manifest alignment,
-   correct video aggregation,
-   correct metric calculations,
-   correct output naming,
-   clean predictions are unchanged when reused,
-   transformed images differ from clean images,
-   transformation metadata is correctly stored.

Only after this validation should the complete nine-checkpoint
experiment be executed.

------------------------------------------------------------------------

## 48. Full experiment execution

The complete experiment is:

``` text
9 checkpoints
    x
enabled transformations
    x
3 severity levels
    x
all usable test frames
```

The clean condition is evaluated once per checkpoint and reused.

For brightness, dark and bright directions are separate conditions.

Optional transformations are only included when their configuration
flags are enabled.

No transformation combinations are part of the default run.

------------------------------------------------------------------------

## 49. Output directory structure

Robustness results should be separated from Approach 1 outputs.

A recommended structure is:

``` text
data/
└── output/
    └── robustness/
        └── <experiment_id>/
            ├── config.json
            ├── config.txt
            ├── clean/
            ├── xception_seed42/
            ├── xception_seed123/
            ├── xception_seed2024/
            ├── efficientnet_b0_seed42/
            ├── efficientnet_b0_seed123/
            ├── efficientnet_b0_seed2024/
            ├── resnet50_seed42/
            ├── resnet50_seed123/
            └── resnet50_seed2024/
```

Within each checkpoint:

``` text
<checkpoint>/
├── clean/
├── jpeg/
├── resize/
├── brightness_dark/
├── brightness_bright/
├── gaussian_noise/
├── gaussian_blur/
├── contrast/
└── crop/
```

Disabled transformations should not produce result directories.

------------------------------------------------------------------------

## 50. Raw result files

Each evaluated condition should retain machine-readable results.

Recommended files include:

``` text
frame_predictions.csv
video_predictions_mean.csv
video_predictions_median.csv
video_predictions_mode.csv
metrics.json
```

The exact names may be adapted to the existing repository conventions.

The frame prediction table is the most important raw artifact because
downstream analysis can be regenerated from it.

------------------------------------------------------------------------

## 51. Aggregate result files

At the experiment level, generate tables such as:

``` text
seed_summary.csv
architecture_summary.csv
transformation_summary.csv
severity_summary.csv
manipulation_summary.csv
confidence_summary.csv
prediction_flip_summary.csv
error_transition_summary.csv
bootstrap_summary.csv
```

A single wide master table may also be generated for convenient
analysis.

------------------------------------------------------------------------

## 52. Figures

The robustness analysis should generate:

``` text
figures/
├── performance_vs_severity/
├── degradation/
├── architecture_comparison/
├── seed_variation/
├── manipulation_comparison/
├── confidence/
├── prediction_flips/
├── confusion_matrices/
└── visual_examples/
```

At minimum, useful plots include:

-   F1 vs severity.
-   ROC-AUC vs severity.
-   Delta F1 vs severity.
-   Delta ROC-AUC vs severity.
-   architecture comparison.
-   seed variation.
-   per-manipulation degradation.
-   prediction-flip rate.
-   confidence shift.
-   confusion-matrix comparisons.

------------------------------------------------------------------------

## 53. Expected experiment matrix

For the three core transformations:

``` text
3 architectures
x 3 seeds
x 3 core transformations
x 3 severity levels
```

plus the clean reference for each checkpoint.

Brightness contains two directions, so it contributes:

``` text
3 architectures
x 3 seeds
x 2 directions
x 3 severity levels
```

Optional transformations add the corresponding three-severity conditions
only when enabled.

The exact number of inference conditions therefore depends on which
optional transformations are activated.

This should be calculated automatically from configuration rather than
hard-coded.

------------------------------------------------------------------------

## 54. Avoiding unnecessary computation

The implementation should avoid redundant inference where possible.

The following should be cached or reused:

-   clean frame predictions,
-   transformed image generation within a condition,
-   shared transformation parameters,
-   aggregate video mappings.

The implementation should not rerun clean inference separately for:

``` text
JPEG
Resize
Brightness
Gaussian noise
...
```

The same clean result is the reference for all comparisons.

------------------------------------------------------------------------

## 55. Data integrity checks

Before evaluating a transformation, verify:

-   frame path exists,
-   video ID is present,
-   category is valid,
-   label is valid,
-   transformation metadata is valid,
-   severity is valid,
-   transformed image can be decoded,
-   transformed image dimensions are valid.

The evaluator should report failed frames rather than silently
discarding them.

A failure summary should include:

``` text
total frames
successful frames
failed frames
failure rate
```

If transformation failures occur, they must not be confused with CNN
robustness failures.

------------------------------------------------------------------------

## 56. Prediction alignment

Clean and transformed predictions must be paired using stable
identifiers.

The preferred key is based on the source frame identity, such as:

``` text
video_id + frame_path
```

or an equivalent unique manifest identifier.

This ensures that:

``` text
clean frame X
```

is compared with:

``` text
JPEG severity 2 version of frame X
```

and not with another frame from the same video.

This alignment is required for:

-   probability shifts,
-   frame prediction flips,
-   paired bootstrap,
-   clean/transformed error transitions.

------------------------------------------------------------------------

## 57. Statistical pairing requirements

For video-level paired analysis:

``` text
video_id
    |
    +--> clean prediction
    +--> transformed prediction
```

Each video contributes one paired observation for the selected
aggregation method.

Frames are not resampled independently for the primary bootstrap.

If a video has missing transformed predictions because of a processing
failure, the analysis must record the exclusion and report the resulting
paired sample size.

------------------------------------------------------------------------

## 58. Interpretation framework

The interpretation should follow:

``` text
claim
  ->
measured result
  ->
comparison with clean baseline
  ->
possible technical interpretation
```

Avoid describing a detector as "robust" solely because its transformed
accuracy remains high.

A robustness claim should consider:

-   absolute transformed performance,
-   degradation from clean,
-   severity dependence,
-   seed variation,
-   manipulation-specific behavior,
-   prediction stability,
-   confidence shifts.

For example:

> The detector retained high F1 under moderate JPEG compression, with a
> small change relative to its clean test result.

is preferable to:

> The detector is highly robust to JPEG compression.

The second statement requires a defined threshold for what qualifies as
"highly robust."

------------------------------------------------------------------------

## 59. Primary reporting hierarchy

The final Approach 2 report should prioritize:

### Primary

-   Clean vs transformed F1.
-   Clean vs transformed ROC-AUC.
-   Delta F1.
-   Delta ROC-AUC.
-   Performance versus severity.
-   Mean ± SD across seeds.
-   Per-manipulation degradation.

### Secondary

-   Accuracy.
-   Precision.
-   Recall.
-   False-positive rate.
-   False-negative rate.
-   Prediction-flip rate.
-   Confidence shift.
-   Confusion matrices.
-   Error-state transitions.
-   Bootstrap confidence intervals.

### Diagnostic

-   Frame-level flip rate.
-   Architecture robustness ranking.
-   Transformation ranking.
-   Area under performance-severity curve.
-   Clean-performance versus robustness relationship.
-   Visual examples.

------------------------------------------------------------------------

## 60. Reproducible experiment naming

Every robustness run should have a unique experiment identifier.

A suitable pattern is:

``` text
robustness_<timestamp>
```

or an equivalent repository convention.

The output must contain a configuration snapshot so that the experiment
can be reconstructed without relying on the current global
configuration.

The experiment identifier should be included in generated reports and
tables.

------------------------------------------------------------------------

## 61. Logging

The robustness runner should log:

-   experiment ID,
-   model,
-   seed,
-   checkpoint,
-   transformation,
-   severity,
-   parameter values,
-   number of frames processed,
-   number of videos processed,
-   processing failures,
-   inference duration,
-   output path.

Progress should be visible during long runs.

The logging level should avoid printing every frame by default.

------------------------------------------------------------------------

## 62. Error handling

A failed transformation on one frame should not crash the entire
robustness experiment.

The evaluator should:

1.  record the failure,
2.  identify the affected frame,
3.  continue where safe,
4.  include the failure count in the final summary.

However, a systemic failure affecting many frames must be surfaced
clearly and should stop the affected condition rather than producing
misleading metrics.

------------------------------------------------------------------------

## 63. Performance considerations

The experiment is evaluation-only, but the test set contains many face
crops.

The implementation should therefore:

-   use batched inference,
-   reuse the existing DataLoader strategy where practical,
-   keep transformations in memory,
-   avoid writing every transformed image to disk,
-   use GPU inference when available,
-   use inference/no-gradient mode,
-   avoid recomputing clean predictions,
-   preserve the existing automatic batch-size behavior where
    compatible.

The robustness layer must not introduce unnecessary CPU-to-GPU
transfers.

------------------------------------------------------------------------

## 64. Reuse of existing checkpoints

Approach 2 must locate the actual Approach 1 checkpoints rather than
retraining models.

The evaluator should support explicit checkpoint paths and, where
repository conventions permit, discovery of the nine baseline
checkpoints.

A checkpoint must be identified by:

``` text
architecture
seed
checkpoint path
```

The experiment configuration should record all three.

------------------------------------------------------------------------

## 65. Reuse of existing manifest

The test manifest is the authoritative source for:

-   frame paths,
-   video IDs,
-   categories,
-   labels,
-   split membership.

Approach 2 must not reconstruct labels from directory names if the
manifest already provides the required metadata.

This reduces the risk of disagreement between the baseline and
robustness experiments.

------------------------------------------------------------------------

## 66. Testing the clean-equivalence condition

The robustness implementation should include a test ensuring that the
clean path produces the same model inputs and predictions as the
existing Approach 1 evaluation path.

This is critical.

A robustness implementation must not accidentally alter baseline
performance through:

-   different normalization,
-   different resizing,
-   different image decoding,
-   different thresholding,
-   different aggregation,
-   different frame selection.

The clean condition should therefore be treated as a regression test
against Approach 1.

------------------------------------------------------------------------

## 67. Approach 2 acceptance criteria

Implementation is considered complete only when all of the following are
satisfied:

-   All nine Approach 1 checkpoints can be evaluated.
-   Clean predictions reproduce the expected baseline evaluation path.
-   JPEG compression works at three severity levels.
-   Lower-resolution resizing works at three severity levels.
-   Darkening works at three severity levels.
-   Brightening works at three severity levels.
-   Gaussian noise is independently configurable.
-   Gaussian blur is independently configurable.
-   Contrast is independently configurable.
-   Centered cropping is independently configurable.
-   Optional transformations default to disabled.
-   Transformations are applied before model-specific resize.
-   The same corruption is supplied to all architectures.
-   Clean inference is computed once per checkpoint and reused.
-   Mean video aggregation remains primary.
-   Median and mode remain secondary.
-   Frame-level probabilities are retained.
-   Video-level metrics are generated.
-   Per-manipulation metrics are generated.
-   Delta metrics are generated relative to clean.
-   Seed mean ± SD is generated.
-   Prediction-flip analysis is generated.
-   Confidence-shift analysis is generated.
-   Error-state transition analysis is generated.
-   Video-level paired bootstrap is implemented.
-   Visual candidate examples are generated.
-   Configuration is saved with results.
-   Transformation failures are logged.
-   Results are reproducible from the recorded configuration.

------------------------------------------------------------------------

## 68. Implementation sequence

### Phase 1: repository and baseline integration

-   Confirm the current `main` implementation.
-   Confirm all Approach 1 checkpoints.
-   Confirm test manifest format.
-   Confirm model-specific preprocessing.
-   Confirm existing evaluation utilities.
-   Confirm output naming conventions.

### Phase 2: robustness configuration

-   Add robustness configuration.
-   Add transformation enable flags.
-   Add severity parameters.
-   Add validation for parameter ranges.
-   Add configuration serialization.

### Phase 3: transformation module

Implement:

1.  JPEG compression.
2.  Lower-resolution resizing.
3.  Brightness.
4.  Gaussian noise.
5.  Gaussian blur.
6.  Contrast.
7.  Centered crop.

Add deterministic behavior and unit tests.

### Phase 4: inference integration

-   Add robustness transformation to the inference path.
-   Preserve model-specific resize.
-   Preserve normalization.
-   Preserve thresholding.
-   Preserve aggregation.
-   Store transformation metadata.

### Phase 5: clean regression test

-   Run one checkpoint.
-   Compare clean results with Approach 1.
-   Resolve any discrepancy before robustness inference.

### Phase 6: pilot robustness experiment

Run one checkpoint with:

-   clean,
-   JPEG,
-   resize,
-   darkening,
-   brightening.

Inspect raw predictions, metrics, plots, and visual examples.

### Phase 7: complete core experiment

Run all nine checkpoints for:

-   JPEG,
-   resize,
-   brightness dark,
-   brightness bright.

### Phase 8: optional transformations

Enable and run independently:

-   Gaussian noise,
-   Gaussian blur,
-   contrast,
-   centered crop.

### Phase 9: statistical analysis

Generate:

-   seed-level statistics,
-   architecture-level statistics,
-   paired bootstrap confidence intervals,
-   manipulation-level results,
-   confidence analysis,
-   prediction-flip analysis,
-   error-state transitions.

### Phase 10: reporting

Generate:

-   summary tables,
-   severity curves,
-   degradation plots,
-   architecture comparisons,
-   manipulation comparisons,
-   visual examples,
-   Approach 2 report.

------------------------------------------------------------------------

## 69. Planned documentation files

The main design document is:

``` text
docs/plans/Approach_02_plan.md
```

The experimental results should be documented separately:

``` text
data/reports/approach-2-robustness-report.md
```

If required, a concise implementation summary can also be added after
completion.

The plan document describes what will be implemented.

The report describes what was actually observed.

These should not be mixed.

------------------------------------------------------------------------

## 70. Final experiment flow

The complete Approach 2 workflow is:

``` text
Approach 1 test manifest
        |
        v
Existing saved face crops
        |
        +----------------------+
        |                      |
        v                      v
      CLEAN              TRANSFORMATION
        |                      |
        |             +--------+--------+
        |             |        |        |
        |           JPEG     Resize  Brightness
        |             |        |       /     \
        |          severity severity Dark   Bright
        |                      |       |       |
        |             +--------+--------+-------+
        |             |
        |       Optional transformations
        |       when explicitly enabled
        |             |
        +-------------+
                      |
                      v
            Model-specific resize
                      |
                      v
               Normalization
                      |
                      v
             CNN checkpoint
                      |
                      v
             Frame probabilities
                      |
                      v
            Video-level aggregation
                      |
                      v
          Clean vs transformed analysis
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Metrics     Stability   Confidence
          |           |           |
          +-----------+-----------+
                      |
                      v
             Statistical analysis
                      |
                      v
             Architecture analysis
                      |
                      v
             Manipulation analysis
                      |
                      v
              Visual verification
                      |
                      v
             Approach 2 report
```

------------------------------------------------------------------------

## 71. Research outputs

Approach 2 should ultimately provide evidence for the following
questions:

1.  How much does each CNN's performance change under JPEG compression?
2.  How much does performance change under reduced spatial resolution?
3.  Does darkening affect the detector differently from brightening?
4.  Which optional corruption types cause the largest performance
    degradation when enabled?
5.  Does degradation increase monotonically with severity?
6.  Do the three CNN architectures exhibit different robustness
    profiles?
7.  Is robustness consistent across the three training seeds?
8.  Are some manipulation categories more sensitive to particular
    transformations?
9.  Do transformations change false-positive and false-negative behavior
    differently?
10. How often do video-level decisions flip relative to clean inputs?
11. Does the model's fake probability shift even when its binary
    decision remains unchanged?
12. How often do transformations convert a correct clean prediction into
    an incorrect prediction?
13. Are transformations ever beneficial for videos that were incorrectly
    classified under clean conditions?
14. Are the observed robustness differences consistent across seeds?
15. Does stronger clean performance correspond to lower or higher
    degradation under corruption?

These analyses are all derived from the same controlled evaluation
framework and do not require additional model training.

------------------------------------------------------------------------

## 72. Constraints and methodological safeguards

The following safeguards are mandatory:

-   Do not train on transformed test images.
-   Do not tune model parameters using transformed test performance.
-   Do not change the test split.
-   Do not change the binary classification target.
-   Do not apply different corruption parameters to different
    architectures.
-   Do not use random corruption without deterministic control.
-   Do not treat frames as independent videos in primary statistical
    analysis.
-   Do not interpret prediction stability as equivalent to robustness.
-   Do not rank transformations solely by a single metric.
-   Do not call a model robust without reporting degradation relative to
    clean.
-   Do not combine transformations in the default experiment.
-   Do not allow optional transformations to become enabled implicitly.
-   Do not alter model-specific input resolution as part of the resize
    robustness experiment.
-   Do not modify model weights.

------------------------------------------------------------------------

## 73. Reproducibility checklist

Before considering an Approach 2 result final, verify:

``` text
[ ] Correct Git commit recorded
[ ] Correct checkpoint recorded
[ ] Correct seed recorded
[ ] Correct test manifest recorded
[ ] Clean condition evaluated
[ ] Clean predictions cached
[ ] Transformation recorded
[ ] Severity recorded
[ ] Exact parameter recorded
[ ] Random seed recorded where applicable
[ ] Frame predictions saved
[ ] Video predictions saved
[ ] Metrics saved
[ ] Failed-frame count recorded
[ ] Per-manipulation results generated
[ ] Seed aggregation generated
[ ] Confidence analysis generated
[ ] Prediction-flip analysis generated
[ ] Error-state analysis generated
[ ] Bootstrap analysis generated
[ ] Visual examples verified
[ ] Configuration snapshot saved
```

------------------------------------------------------------------------

## 74. Scope boundary for later approaches

Approach 2 produces robustness evidence that can be reused by later
approaches.

Approach 3 can use selected clean and transformed examples for Grad-CAM
analysis.

This allows the project to investigate whether the visual evidence used
by the detector changes when its input is corrupted.

Approach 4 can use the baseline and robustness results as reference
points when evaluating generalization to unseen manipulation techniques.

Approach 2 itself should not incorporate those later experiments.
