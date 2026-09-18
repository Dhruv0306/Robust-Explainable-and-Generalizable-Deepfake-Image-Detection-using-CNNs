# Approach 2 novelty plan

## 1. Purpose

This plan extends the completed Approach 2 robustness experiment with three analysis components:

1. Confidence-to-decision stability.
2. Cross-architecture failure agreement.
3. Transformation × manipulation vulnerability.

The extension is evaluation-only. It does not introduce new model training, fine-tuning, corruption types, severity values, dataset splits, or checkpoints.

The analysis uses the existing Approach 2 experiment: the existing FaceForensics++ C23 test population, 24 test videos, Xception/EfficientNet-B0/ResNet50, seeds 42/123/2024, nine checkpoints, and the existing 117-condition matrix.

## 2. Repository audit before implementation

The `approach/2-Robustness` branch already describes confidence shifts, prediction flips, error-state transitions, and manipulation-level analysis in its plan/report.

However, the current `src/robustness_analysis.py` fetched from the branch primarily performs seed aggregation, architecture aggregation, severity plots, and report generation. Therefore, these analyses must be verified against the actual generated artifacts before adding duplicate code.

The cross-architecture failure-agreement analysis is the clearly missing component.

Implementation will therefore distinguish between:

- existing analysis that can be reused,
- existing analysis that needs completion or consolidation,
- genuinely new cross-architecture analysis.

## 3. Fixed experimental scope

Do not change the completed robustness experiment.

Use exactly the existing Approach 2 test population, test manifest, nine checkpoints, and 117 conditions.

The conditions remain:

- clean
- JPEG Q=80, 50, 20
- resize s=0.75, 0.50, 0.25
- darkening alpha=0.80, 0.60, 0.40
- brightening alpha=1.20, 1.40, 1.60

No new corruption types or transformation combinations are introduced.

No CNN retraining or fine-tuning is performed.

## 4. Phase 0 - verify existing outputs

Before implementing new calculations:

### 4.1 Master summary

Verify `master_summary.csv` contains the expected checkpoint-condition combinations and fields for model, seed, transformation, severity, parameter, video-level metrics, and delta metrics.

### 4.2 Raw predictions

Verify that the condition-level prediction files contain:

- video ID,
- frame ID/path,
- category,
- ground-truth label,
- fake probability,
- binary prediction.

Use these prediction files as the source for the novelty analyses wherever possible.

### 4.3 Clean reference

Verify that each transformed condition is paired with the same checkpoint's cached clean prediction. Do not perform unnecessary second clean inference.

### 4.4 Existing confidence and decision analysis

Check whether confidence summaries, probability shifts, prediction flips, and error-state transitions are actually generated. Reuse correct implementations and extend incomplete ones.

### 4.5 Existing manipulation analysis

Verify manipulation-level metrics from the prediction metadata and determine the exact number of videos contributing to each category. Do not assume category counts from the report.

## 5. Novelty A - confidence-to-decision stability

### Research question

How does corruption alter the detector's continuous fake probability before and after the binary decision changes?

The novelty claim should not be that confidence shift itself has never been studied. The contribution is the joint analysis of continuous confidence movement, binary decision stability, and error-state transitions at the video level.

### Measures

For every architecture, seed, video, transformation, and severity:

- clean mean fake probability,
- transformed mean fake probability,
- `delta_p_fake = transformed_probability - clean_probability`,
- absolute probability shift,
- clean binary prediction,
- transformed binary prediction,
- clean correctness,
- transformed correctness.

Probability drift must be explicitly stratified by ground-truth class:

- **Ground-truth Fake:** a negative `delta_p_fake` represents movement toward the Real class and therefore confidence erosion for detecting manipulated content. This is directly relevant to false-negative risk.
- **Ground-truth Real:** a negative `delta_p_fake` represents movement further away from the Fake decision threshold and therefore increased separation from a false-positive decision. A positive shift toward 0.5 or beyond can instead indicate increasing false-positive risk.

Report the two subsets separately rather than combining their probability shifts into one class-agnostic value.

Report mean and median shifts, with standard deviation where appropriate.

Separate results by all videos, ground-truth Real/Fake class, and clean-correct versus clean-incorrect videos where useful.

### Decision stability

Use the existing 0.5 threshold.

- same clean/transformed prediction -> stable decision
- different prediction -> prediction flip

Report flip count, flip rate, and flip direction where useful.

### Error transitions

Use the existing four states:

| Clean | Transformed | State |
|---|---|---|
| Correct | Correct | Stable correct |
| Correct | Incorrect | Robustness failure |
| Incorrect | Correct | Transformation correction |
| Incorrect | Incorrect | Persistent error |

This separates confidence movement from actual decision failure.

### Outputs

Create or verify:

```text
confidence_decision_summary.csv
prediction_flip_summary.csv
error_transition_summary.csv
```

### Figures

Generate:

1. Fake-probability shift versus severity.
2. Prediction flip rate versus severity.
3. Error-state transition distribution versus severity.

Do not introduce an arbitrary threshold for a "large" confidence shift without methodological justification.

## 6. Novelty B - cross-architecture failure agreement

This is the primary genuinely new analysis.

### Research question

When a corruption causes a detector to fail, do different CNN architectures fail on the same videos or on architecture-specific videos?

### Comparison unit

The basic unit is:

```text
same video
same transformation
same severity
same seed
different architecture
```

Perform both:

1. seed-level pairwise comparisons,
2. architecture-level aggregation across the three seeds.

Architecture-level aggregation is the main summary, while seed-level results show consistency.

### Robustness-failure definition

For the main failure-agreement analysis:

```text
clean correct -> transformed incorrect
```

Define:

```text
FailureSet(A,T,S)
```

as the videos for which architecture/checkpoint A is correct under clean input but incorrect under the transformed condition.

This avoids treating persistent clean errors as corruption-induced failures.

### Pairwise failure overlap

For architectures A and B:

```text
Jaccard(A,B) =
|FailureSet(A) ∩ FailureSet(B)|
/
|FailureSet(A) ∪ FailureSet(B)|
```

High overlap indicates shared failure cases. Low overlap indicates more architecture-specific failure cases.

Define the empty-union case explicitly:

```text
J(A,B) = 1.0, if |FA ∪ FB| = 0
         |FA ∩ FB| / |FA ∪ FB|, otherwise
```

An empty union means neither architecture has a corruption-induced failure under that paired condition, so the architectures have complete agreement on the absence of failure. The direct disagreement rate remains a separate metric.

### Prediction disagreement

Also calculate:

```text
Disagreement(A,B) =
videos where prediction_A != prediction_B
/
paired videos
```

This is broader than robustness-failure overlap and must not be interpreted as the same quantity.

### Three-architecture failure consensus

For every video-condition, count the number of architectures experiencing a robustness failure:

```text
0, 1, 2, or 3 architectures
```

This identifies isolated versus shared failures.

### Outputs

Create:

```text
architecture_disagreement_summary.csv
failure_overlap_summary.csv
failure_consensus_summary.csv
```

Retain transformation, severity, architecture pair, seed where applicable, paired-video count, disagreement rate, failure-set sizes, intersection, union, and Jaccard overlap.

### Figures

Generate:

1. Pairwise prediction-disagreement heatmap.
2. Pairwise robustness-failure Jaccard heatmap.
3. Distribution of videos with 0/1/2/3 architectures failing.

Do not collapse the analysis into a single architecture robustness score.

## 7. Novelty C - transformation × manipulation vulnerability

### Research question

Does the effect of a corruption depend on the manipulation type being detected?

### Categories

Use the existing fake categories:

- DeepFake
- Face2Face
- FaceSwap
- NeuralTextures

Retain Original/Real samples for supporting false-positive analysis, but the primary vulnerability analysis concerns manipulated categories.

### Primary metric

Use:

```text
Delta F1 = F1_transformed - F1_clean
```

Report both transformed F1 and degradation from the same category's clean condition.

### Secondary metric

Use fake-class recall:

- clean recall,
- transformed recall,
- recall change.

This helps distinguish overall classification degradation from reduced detection of manipulated samples.

### Small-sample safeguards

The current Approach 2 test population contains 24 videos: 12 Real videos and 12 Fake videos distributed across the four manipulation categories. The exact per-category video count must be read from the final prediction metadata and recorded in the output rather than inferred from frame counts.

For every category:

- report the exact video count `N` directly in the summary and heatmap annotation,
- retain seed-level values,
- use mean ± SD where appropriate,
- use paired video-level uncertainty analysis where meaningful,
- never treat individual frames as independent videos.

Because each manipulation category contains only a small number of test videos, category-level results are empirical sensitivity patterns. They should not be presented as strong statistical evidence for a general property of that manipulation type. Frame counts must not be used to imply a larger independent sample size.

### Main matrix

Construct:

```text
                    JPEG    Resize    Darken    Brighten
DeepFake
Face2Face
FaceSwap
NeuralTextures
```

The primary cell value is ΔF1 for a defined severity. A corresponding recall-change matrix can be produced.

For presentation, severity 3 can be used as the strongest-corruption summary, while full severity results remain in the report.

### Outputs

Create or verify:

```text
manipulation_vulnerability_summary.csv
manipulation_f1_delta.csv
manipulation_recall_delta.csv
```

### Figures

Generate:

1. Transformation × manipulation ΔF1 heatmap.
2. Transformation × manipulation recall-change heatmap.
3. Optional category-level severity curves where sample size supports them.

Interpret category-specific degradation as an empirical sensitivity pattern. Do not infer the underlying forensic cue without additional evidence.

## 8. Integrated analysis

The final robustness analysis should connect:

```text
Performance degradation
        |
        +--> Confidence shift
        |
        +--> Decision flips
        |
        +--> Error-state transitions
        |
        +--> Shared vs architecture-specific failures
        |
        +--> Manipulation-specific vulnerability
```

This gives three complementary levels:

### Model-level

F1, ROC-AUC, Accuracy, ΔF1, and ΔROC-AUC.

### Decision-level

Fake-probability shift, prediction flip rate, and error-state transitions.

### Failure-structure level

Architecture disagreement, failure overlap, failure consensus, manipulation-specific ΔF1, and recall change.

These must remain separate measures. No universal robustness score should be introduced.

## 9. Implementation structure

Use a single post-processing module for the novelty extension:

```text
src/
└── robustness_novelty.py
```

This module should load the existing `master_summary.csv` and condition-level prediction artifacts, compute Novelties A, B, and C, and export the summary CSV files and figures.

Unit tests should be added separately:

```text
tests/
└── test_robustness_novelty.py
```

The tests should validate at least:

- expected table schemas,
- Jaccard values within `[0, 1]`,
- correct handling of empty failure-set unions,
- conservation of paired video counts,
- correct clean/transformed video pairing,
- class-stratified probability-shift calculations.

Suggested functions inside `robustness_novelty.py`:

```python
compute_confidence_decision_analysis(...)
compute_prediction_flip_analysis(...)
compute_error_transition_analysis(...)

build_failure_sets(...)
compute_pairwise_architecture_disagreement(...)
compute_pairwise_failure_overlap(...)
compute_architecture_failure_consensus(...)

compute_manipulation_vulnerability(...)
compute_manipulation_recall_change(...)
```

This structure keeps the novelty extension isolated from the existing baseline robustness analysis and avoids unnecessary modification of `robustness_analysis.py`.

## 10. Validation

### Data integrity

- All nine checkpoints are represented.
- All expected conditions are represented.
- No duplicate video-condition-architecture rows.
- Architecture comparisons use identical video IDs.
- Clean/transformed records are correctly paired.
- Category metadata are preserved.

### Metric integrity

- ΔF1 uses the same clean reference as Approach 2.
- Binary decisions use the existing threshold.
- Robustness failure means clean-correct → transformed-incorrect.
- Jaccard uses paired failure sets.
- Empty unions produce NA.
- Manipulation metrics use video-level pairing.

### Reproducibility

- Re-running the analysis from the same prediction files produces identical outputs.
- Existing Approach 2 F1 and ROC-AUC results remain unchanged.
- No checkpoint or raw prediction is modified.

## 11. Report integration

Recommended Approach 2 report structure:

```text
1. Experimental setup
2. Overall robustness results
3. Confidence-to-decision stability
4. Cross-architecture failure agreement
5. Transformation × manipulation vulnerability
6. Integrated robustness interpretation
7. Limitations
8. Output artifacts
```

The contribution should be framed as a diagnostic extension of the robustness experiment, not as a new CNN or robustness-training method.

## 12. Phase B presentation

Use three main figures:

### Novelty 1

Confidence and decision stability:

```text
confidence shift + prediction flip rate
```

### Novelty 2

Cross-architecture failure agreement:

```text
pairwise failure-overlap heatmap
```

with a compact 0/1/2/3 architecture failure distribution if space permits.

### Novelty 3

Transformation × manipulation vulnerability:

```text
ΔF1 heatmap
```

across manipulation categories and transformation types.

For degradation heatmaps, use a diverging color map centered at zero, such as `coolwarm` or `RdBu`. The visual convention should make negative ΔF1 values, representing performance degradation, visually distinct from zero change and positive values. White or a neutral midpoint should represent approximately zero change. Each manipulation cell must also display the contributing video count, `N`.

The same centered-at-zero convention should be used for other change/degradation heatmaps where applicable.

Each figure should state the research question and what is measured rather than listing every metric.

## 13. Implementation order

1. Audit existing Approach 2 prediction and summary artifacts.
2. Verify existing confidence, flip, error-transition, and manipulation analyses.
3. Reuse valid existing calculations.
4. Implement cross-architecture failure agreement.
5. Complete transformation × manipulation vulnerability.
6. Complete or consolidate confidence-to-decision analysis.
7. Generate novelty-specific CSV outputs.
8. Generate novelty-specific figures.
9. Run validation checks.
10. Update the Approach 2 report.
11. Select presentation figures.
12. Commit changes to `approach/2-Robustness`.

## 14. Claims to avoid

Do not claim:

- the corruption algorithms themselves are novel,
- confidence-shift analysis has never appeared in prior literature,
- one architecture is universally more robust,
- stable predictions imply complete robustness,
- category-specific degradation proves a particular forensic feature is used,
- the 24-video test population establishes behavior for all FaceForensics++ data,
- causal mechanisms are proven without additional evidence.

Any literature novelty claim must be supported by a targeted literature review with verified references.

## 15. Success criteria

The extension is complete when:

- all three analyses are reproducible from existing Approach 2 predictions,
- no model retraining is required,
- all nine checkpoints are represented,
- the same 24-video population is used,
- architecture comparisons are paired by video and condition,
- manipulation category counts are verified,
- confidence and decision behavior are separated,
- robustness failures are separated from persistent clean errors,
- shared and architecture-specific failures are quantified,
- transformation-specific manipulation vulnerability is quantified,
- existing Approach 2 baseline results remain unchanged,
- the resulting figures can be explained clearly in the Phase B presentation.
