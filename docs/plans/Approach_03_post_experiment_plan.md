# Approach 3 post-experiment analysis plan

## Current experimental state

The full Approach 3 experiment has completed for all three architectures:

| Model | Seed | Frame samples | Conditions | Frame-condition evaluations |
|---|---:|---:|---:|---:|
| Xception | 2024 | 6,304 | 13 | 81,952 |
| EfficientNet-B0 | 2024 | 6,304 | 13 | 81,952 |
| ResNet50 | 123 | 6,304 | 13 | 81,952 |

The generated outputs are committed under:

```text
data/output/explainability_corrected_v1/
    xception/seed_2024/
    efficientnet_b0/seed_2024/
    resnet50/seed_123/
```

The expensive Grad-CAM generation is therefore complete. The next stage is post-experiment validation and analysis. No further full Grad-CAM run should be performed unless a data-generation defect is discovered.

## 1. Immediate validation

Run the output validator before interpreting results.

The validator must check:

- exactly 6,304 frame samples per model;
- exactly 13 conditions;
- 81,952 frame-condition evaluations per model;
- unique evaluation keys:
  `category, video_id, original_frame_number, transformation`;
- complete condition coverage;
- no failed frames hidden by missing rows;
- no duplicate video-level keys;
- valid source-video counts;
- finite values for core metrics where they are expected;
- consistency of clean and transformed prediction fields.

The expected manipulated population is 60 category-video units, corresponding to 15 source-video identities per manipulation category.

## 2. Statistical-unit validation

The implementation now collapses the four manipulation-category views to the source-video level before inferential testing.

This is the intended inferential unit:

```text
category-level video results
        |
        v
average across manipulation categories
        |
        v
source-video result
        |
        v
paired inference
```

The source-video aggregation should therefore produce 12 source-video observations for the current manipulated test set.

This must be checked explicitly.

## 3. Important statistical correction

The generated `video_level_results.csv` does not contain a clean `ES_cos` value. `ES_cos` is defined only for a clean/transformed pair.

Therefore, the current statistical code must not perform:

```text
clean ES_cos vs transformed ES_cos
```

with a paired Wilcoxon test.

The correct stability analysis is:

```text
ES_cos(clean, transformed)
```

as the direct paired explanation-stability measurement already stored for each transformed condition.

The same applies to `explanation_IoU`.

For each transformation, report:

- mean ES_cos;
- median ES_cos;
- bootstrap 95% CI if required;
- proportion below the predefined stability threshold;
- explanation IoU;
- distribution by architecture and transformation.

The continuous ES_cos remains primary.

## 4. Prediction-state analysis

For every transformed condition, calculate the four prediction states:

```text
Correct -> Correct
Correct -> Incorrect
Incorrect -> Correct
Incorrect -> Incorrect
```

Use video-level source observations.

Report:

- count;
- proportion;
- SO;
- IoU;
- saliency mass;
- hit rate;
- ES_cos;
- explanation IoU;
- faithfulness.

Do not infer causal mechanisms from these states.

A useful descriptive observation is whether high explanation similarity can coexist with low localization agreement. The current Xception outputs already contain examples of this type, but the full analysis must quantify it rather than relying on examples.

## 5. Transformation analysis

For each architecture and transformation:

- clean SO;
- transformed SO;
- clean IoU;
- transformed IoU;
- clean saliency mass;
- transformed saliency mass;
- clean faithfulness;
- transformed faithfulness;
- ES_cos;
- explanation IoU.

Calculate:

```text
D_SO       = SO_clean - SO_transformed
D_IoU      = IoU_clean - IoU_transformed
D_stability = 1 - ES_cos
D_faith    = faithfulness_clean - faithfulness_transformed
```

Do not assume that severity produces monotonic degradation.

## 6. Severity analysis

Analyze each transformation family separately:

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

For each family, calculate the relationship between ordered severity and:

- SO;
- IoU;
- saliency mass;
- faithfulness;
- ES_cos;
- explanation IoU.

Use Spearman correlation as a descriptive/inferential ordered-severity analysis.

Do not describe a relationship as monotonic unless the observed data support it.

## 7. Localization versus faithfulness

Calculate Spearman correlations between:

```text
SO <-> faithfulness_blur
IoU <-> faithfulness_blur
saliency_mass <-> faithfulness_blur
```

Primary faithfulness method:

```text
blur
```

Zero and mean masking are sensitivity analyses.

Stratify where sample size permits:

- architecture;
- manipulation category;
- correct fake;
- missed fake;
- clean;
- transformed.

Because the current source-video aggregation averages categories, category-stratified analysis must be explicitly labelled as category-level descriptive analysis rather than independent source-video inference.

## 8. Detector degradation versus explanation degradation

Approach 3 should use Approach 2's video-level F1 degradation:

```text
D_F1 = F1_clean - F1_transformed
```

and compare it with:

```text
D_SO
D_IoU
D_stability
D_faith
```

This is distinct from:

```text
D_prob = P_fake_clean - P_fake_transformed
```

`D_prob` is a confidence/probability shift, not detector performance loss.

Use Spearman correlation for the relationship analysis.

Do not interpret correlation as causation.

## 9. Architecture comparison

Compare the three architectures descriptively across:

- clean localization;
- transformed localization;
- faithfulness;
- explanation stability;
- prediction-state behavior;
- severity behavior.

Use confidence intervals and paired tests where the statistical design supports them.

Do not produce an overall architecture ranking.

## 10. Global explanation analysis

Retain:

- overall Grad-CAM heatmap;
- category heatmaps;
- within-video consistency;
- within-category consistency;
- saliency entropy.

Treat heatmaps as visual evidence, not independent quantitative observations.

Do not label high or low entropy as inherently better.

## 11. Representative cases

Use predefined selection rules for:

- correct fake;
- missed fake;
- highest SO;
- lowest SO;
- prediction-preserved transformation;
- prediction-changed transformation.

Avoid selecting cases after seeing which examples support a preferred interpretation.

## 12. Statistical reporting

For every inferential result, retain:

```text
n_total
n_valid
n_nonzero
effect size where applicable
raw p
adjusted p
95% CI where applicable
test_status
```

Benjamini-Hochberg correction remains separated into:

- family 1: localization;
- family 2: faithfulness;
- family 3: explanation stability;
- family 4a: localization-faithfulness relationships;
- family 4b: detector-explanation degradation relationships.

## 13. Recommended execution order

```text
1. Validate all three completed outputs
2. Validate source-video aggregation
3. Correct stability statistical handling
4. Recompute statistical outputs without rerunning Grad-CAM
5. Run post-hoc descriptive analysis
6. Integrate Approach 2 F1 degradation
7. Generate final tables and figures
8. Audit interpretation
9. Update Approach 3 report
10. Prepare Phase B presentation material
```

The priority is to validate the generated evidence before writing claims about robustness, faithfulness, or explanation behavior.
