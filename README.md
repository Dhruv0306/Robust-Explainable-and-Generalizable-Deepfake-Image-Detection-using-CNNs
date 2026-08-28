# Robust, Explainable, and Generalizable Deepfake Image Detection using CNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Project](https://img.shields.io/badge/Project-Research-blue.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Deepfake%20Detection-red.svg)]()

> **Advanced Data Mining - Research Project**
> **Course:** CS G520
> **Institution:** BITS Pilani, Goa Campus
> **Date:** August 2026

---

## Overview

Deepfake image generation has progressed from relatively obvious face manipulations to increasingly realistic synthetic and edited images. Although modern deepfake detectors can achieve strong performance on benchmark datasets, their performance may degrade when the input distribution changes.

A detector that performs well on clean benchmark images may fail when images are:

* JPEG compressed
* Resized
* Blurred
* Corrupted by noise
* Altered in brightness or contrast
* Produced using manipulation techniques not observed during training
* Drawn from a different dataset or distribution

A second challenge is **interpretability**. A prediction such as `Fake = 0.97` does not explain what visual evidence influenced the model. This project therefore investigates not only whether a CNN can classify images correctly, but also **how robustly it performs, what visual evidence it uses, and whether that evidence transfers to unseen manipulation distributions**.

The project is intentionally designed as an **empirical evaluation study**. Rather than proposing a new state-of-the-art deepfake architecture, the same CNN-based detector is evaluated under controlled experimental conditions across three research dimensions:

> **Robustness → Explainability → Generalizability**

---

## Research Question

> **Can a CNN-based deepfake image detector accurately distinguish real and manipulated images while remaining robust to common image transformations, generalizing to unseen manipulation techniques, and providing meaningful visual explanations of its predictions?**

---

## Research Objectives

The project has four primary objectives:

1. **Establish a reproducible CNN baseline** for real-versus-manipulated image classification.
2. **Quantify robustness** under common image transformations and degradation conditions.
3. **Analyse model explanations** using Grad-CAM and determine whether the highlighted regions provide meaningful and stable evidence.
4. **Evaluate generalizability** on manipulation techniques and datasets that are not represented during training.

The study additionally aims to understand where the detector succeeds, where it fails, and how robustness, explainability, and generalization interact.

---

# Research Framework

The experimental framework consists of four cumulative approaches.

```text
                    ┌──────────────────────────┐
                    │   CNN Baseline Detector  │
                    │       Approach 1         │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
       ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
       │   Robustness   │ │ Explainability│ │  Generalizability│
       │   Approach 2   │ │  Approach 3   │ │    Approach 4    │
       └────────────────┘ └───────────────┘ └──────────────────┘
                │                │                │
                ▼                ▼                ▼
       Image transformations   Grad-CAM      Unseen manipulations
       and degradation         analysis      / datasets
```

The same baseline model is retained across the approaches wherever possible. This allows changes in performance to be attributed to the experimental condition rather than to architectural changes.

---

# Approaches

## Approach 1 - Existing CNN Baseline

The first approach establishes the control condition for the entire study.

An existing CNN architecture will be selected based on:

* Availability of a reliable implementation
* Compatibility with the selected dataset
* Computational requirements
* Suitability for image-based deepfake detection

Candidate architectures include:

* Xception
* EfficientNet
* ResNet
* Other lightweight CNN architectures where appropriate

The final architecture will be selected after evaluating dataset and computational constraints.

### Baseline Evaluation

The baseline will use a fixed preprocessing and evaluation protocol.

Where possible, dataset splitting will be performed at the subject or video level to reduce identity leakage.

The baseline evaluation will report:

| Metric           | Purpose                              |
| ---------------- | ------------------------------------ |
| Accuracy         | Overall classification correctness   |
| Precision        | False-positive behaviour             |
| Recall           | Ability to detect manipulated images |
| F1-score         | Balance between precision and recall |
| ROC-AUC          | Threshold-independent discrimination |
| PR-AUC           | Useful under class imbalance         |
| Confusion Matrix | Detailed error analysis              |

The baseline serves as the reference point for all subsequent experiments.

---

## Approach 2 - Robustness Under Image Transformations

The second approach investigates how the CNN behaves when image quality or appearance is modified.

The model itself remains unchanged. Only the test images are transformed.

### Planned Transformations

* JPEG compression
* Image resizing
* Gaussian blur
* Gaussian noise
* Brightness changes
* Contrast changes
* Controlled cropping

Where feasible, transformations will be evaluated at multiple severity levels.

For example:

```text
Clean Image
    │
    ├── JPEG Quality: High ───────► Evaluation
    ├── JPEG Quality: Medium ─────► Evaluation
    ├── JPEG Quality: Low ────────► Evaluation
    │
    ├── Blur: Low ────────────────► Evaluation
    ├── Blur: Medium ─────────────► Evaluation
    ├── Blur: High ───────────────► Evaluation
    │
    └── Noise: Low/Medium/High ───► Evaluation
```

### Robustness Metrics

In addition to standard classification metrics, the study will measure performance degradation relative to the clean test set.

For a metric \(M\):

$$
\Delta M = M_{\text{clean}} - M_{\text{transformed}}
$$

This provides a quantitative robustness profile instead of assigning a single qualitative label such as "robust" or "not robust".

The analysis will investigate:

* Which transformations cause the largest degradation?
* Does degradation increase with transformation severity?
* Are real and fake images affected differently?
* Does the model remain calibrated under degraded conditions?
* Do explanation maps change when the input is transformed?

---

## Approach 3 - Explainability with Grad-CAM

The third approach investigates **what the CNN is using to make its decisions**.

Grad-CAM will initially be used as the primary explainability technique because it can map convolutional feature activations back to spatial regions of the input image.

### Explanation Categories

Grad-CAM analysis will be performed on:

* Correct Real predictions
* Correct Fake predictions
* False positives
* False negatives
* Images under controlled transformations
* Representative examples from seen and unseen manipulation distributions

The analysis will examine whether salient regions correspond to:

* Facial regions
* Plausible manipulation boundaries
* Local image artifacts
* Background regions
* Image borders
* Other potentially irrelevant regions

### Explanation Stability

Explanations will also be compared before and after image transformations.

Conceptually:

```text
Original Image
      │
      ├── CNN Prediction
      │
      └── Grad-CAM Map
              │
              ▼
       Visual Evidence
              │
       Transformation
              │
              ▼
Transformed Image
      │
      ├── CNN Prediction
      │
      └── Grad-CAM Map
              │
              ▼
     Explanation Stability
```

If computationally feasible, a quantitative faithfulness experiment will also be performed by masking or perturbing salient regions and measuring the resulting change in prediction confidence.

The underlying idea is:

> If a region is genuinely important to the model's prediction, modifying that region should affect the prediction more than modifying irrelevant regions.

---

## Approach 4 - Generalizability to Unseen Manipulations

The fourth approach evaluates whether the detector has learned transferable forensic evidence or has instead learned cues specific to the training distribution.

### Cross-Manipulation Evaluation

The primary experimental design is **cross-manipulation evaluation**.

If the selected dataset contains multiple manipulation categories, one manipulation category will be excluded entirely from training and validation and reserved for testing.

```text
Training Distribution
├── Manipulation A
├── Manipulation B
└── Manipulation C

             ↓

          CNN Model

             ↓

Unseen Test Distribution
└── Manipulation D
```

The detector is evaluated on the unseen manipulation without retraining.

Where dataset size permits, multiple held-out manipulation experiments may be performed.

### Cross-Dataset Evaluation

A secondary experiment may evaluate cross-dataset generalization:

```text
Train:
FaceForensics++

       ↓
   CNN Detector
       ↓
Test:
Celeb-DF v2
```

No retraining is performed on the target dataset.

The primary quantity of interest is the **generalization gap** between in-domain and unseen-domain performance.

$$
\text{Generalization Gap}
=
M_{\text{in-domain}}
-
M_{\text{unseen}}
$$

The study will also compare Grad-CAM explanations between seen and unseen distributions to investigate whether the model focuses on similar visual evidence.

---

# Research Verticals

The four approaches are organized around three complementary research verticals.

| Vertical             | Core Question                                                     | Main Evaluation                                 |
| -------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| **Robustness**       | Does performance remain stable under realistic image degradation? | Transformation-specific performance drop        |
| **Explainability**   | What visual evidence drives the CNN's predictions?                | Grad-CAM and explanation faithfulness/stability |
| **Generalizability** | Does the detector transfer to unseen manipulation distributions?  | Generalization gap and cross-domain metrics     |

The interaction between these dimensions is also important.

For example:

> A detector may remain accurate after mild JPEG compression while its Grad-CAM maps shift from plausible facial regions to irrelevant image regions.

Similarly:

> A detector may achieve strong in-domain performance but fail on an unseen manipulation while simultaneously producing qualitatively different explanations.

Such cases can provide evidence about the type of cues learned by the detector.

---

# Dataset Strategy

The proposal identifies **FaceForensics++** as the primary dataset candidate and **Celeb-DF v2** as the primary external generalization candidate.

### Primary Dataset - FaceForensics++

FaceForensics++ is preferred because it:

* Contains multiple manipulation methods
* Is widely used in deepfake detection research
* Provides a strong baseline for comparison with existing literature
* Supports controlled manipulation-based evaluation
* Can support robustness experiments using derived image transformations

Resource:

https://github.com/ondyari/FaceForensics

### External Dataset - Celeb-DF v2

Celeb-DF v2 is considered the primary cross-dataset candidate because it contains higher-quality deepfakes designed to reduce obvious synthesis artifacts.

It can be used to evaluate:

```text
Train on FaceForensics++
             ↓
       Fixed CNN Model
             ↓
Evaluate on Celeb-DF v2
             ↓
   Cross-Dataset Gap
```

Resource:

https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics

### Additional Candidates

Depending on computational resources and dataset accessibility:

* **DFDC**
* **DeeperForensics-1.0**

DFDC provides large-scale variation and multiple face-modification algorithms, while DeeperForensics-1.0 is particularly relevant to robustness because it contains realistic perturbations at multiple severity levels.

---

# Evaluation Metrics

The project uses multiple complementary metrics rather than relying on accuracy alone.

| Metric                    | Interpretation                                             |
| ------------------------- | ---------------------------------------------------------- |
| **Accuracy**              | Overall prediction correctness                             |
| **Precision**             | Reliability of Fake predictions                            |
| **Recall**                | Ability to identify actual Fake images                     |
| **F1-score**              | Balance between Precision and Recall                       |
| **ROC-AUC**               | Ranking and discrimination ability                         |
| **PR-AUC**                | Precision-Recall performance, particularly under imbalance |
| **Confusion Matrix**      | TP, TN, FP, and FN analysis                                |
| **Performance Drop**      | Robustness degradation relative to clean data              |
| **Generalization Gap**    | Difference between in-domain and unseen-domain performance |
| **Explanation Stability** | Consistency of salient regions under controlled changes    |
| **Saliency Faithfulness** | Effect of perturbing regions identified as important       |

The primary quantitative comparison will emphasize **F1-score and ROC-AUC**, together with Precision and Recall.

---

# Hypotheses

The project investigates the following hypotheses.

### H1 - Robustness

> Detector performance will degrade under one or more common image transformations, with degradation depending on transformation type and severity.

### H2 - Explainability

> Grad-CAM will identify non-uniform spatial evidence, and explanation quality or stability will vary across correct, incorrect, and transformed cases.

### H3 - Generalizability

> Performance on unseen manipulation techniques will be lower than performance on seen-manipulation test data, demonstrating a measurable generalization gap.

### H4 - Distribution Dependence

> The magnitude of robustness and generalization degradation will vary by manipulation type, indicating that detector behaviour is dependent on the underlying data distribution.

---

# Experimental Design Principles

To ensure that the study remains scientifically meaningful, the following principles will be maintained:

1. **Keep the baseline architecture fixed** across experiments wherever possible.
2. **Use a fixed preprocessing pipeline** unless the experiment explicitly studies preprocessing.
3. **Avoid train-test leakage**, particularly identity or video-level leakage.
4. **Keep the clean test set untouched** for baseline evaluation.
5. **Change one major experimental factor at a time.**
6. **Report multiple metrics**, rather than accuracy alone.
7. **Separate in-domain performance from out-of-domain performance.**
8. **Evaluate explanations rather than only displaying them.**
9. **Report failure cases**, not only successful predictions.
10. **Document experimental settings** to support reproducibility.

---

# Course Alignment

The project follows a four-phase structure aligned with the Advanced Data Mining course requirements.

| Phase       | Course Requirement                                             | Project Implementation                                                    |
| ----------- | -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Phase A** | Problem definition, literature review, dataset, Approaches 1-4 | Research framing, 25-paper review, dataset selection, experimental design |
| **Phase B** | Approaches 1 and 2                                             | CNN baseline and robustness experiments                                   |
| **Phase C** | Approaches 3 and 4                                             | Grad-CAM explainability and generalization experiments                    |
| **Phase D** | Research / White Paper                                         | Synthesis, analysis, limitations, and final research paper                |

The approaches are intentionally cumulative:

```text
Approach 1
Baseline
   │
   ▼
Approach 2
Robustness
   │
   ▼
Approach 3
Explainability
   │
   ▼
Approach 4
Generalizability
   │
   ▼
Final Research Analysis
```

---

# Expected Contribution

This project does **not** claim to solve deepfake detection in general.

The CNN architecture is treated as an **experimental instrument**.

The primary contribution is an organized empirical framework for studying how an existing CNN behaves across:

* Clean benchmark data
* Realistic image transformations
* Explainability analysis
* Unseen manipulation techniques
* Cross-dataset distribution shifts

The final study aims to characterize:

> **Where the detector succeeds, where it fails, what evidence it uses, and whether that evidence remains reliable when the data distribution changes.**

This framing allows the project to investigate practical limitations of deepfake detectors without conflating benchmark accuracy with real-world reliability.

---

# Expected Outputs

The completed project is expected to produce:

### Baseline

* Trained CNN detector
* Classification metrics
* Confusion matrix
* ROC curve
* Precision-Recall analysis

### Robustness

* Transformation-specific test sets
* Severity-wise evaluation
* Performance degradation curves
* Comparison of clean versus transformed predictions

### Explainability

* Grad-CAM visualizations
* Correct and incorrect prediction analysis
* Explanation stability analysis
* Optional saliency faithfulness measurements

### Generalizability

* Cross-manipulation evaluation
* Optional cross-dataset evaluation
* In-domain versus unseen-domain comparison
* Generalization-gap analysis
* Grad-CAM comparison across distributions

### Final Research Output

* Consolidated experimental results
* Error analysis
* Discussion of robustness, explainability, and generalization trade-offs
* Limitations and future research directions
* Research / white paper

---

# Literature Foundation

The research design is motivated by literature covering:

* General face-forgery detection
* Frequency-domain forensic cues
* Dataset and manipulation shifts
* Robustness to compression and image degradation
* Attention and localization
* Explainable AI for deepfake detection
* Cross-dataset generalization
* Unseen-manipulation detection
* Domain generalization
* Quantitative evaluation of explanations

Representative works include:

1. **Face X-Ray for More General Face Forgery Detection** - Li et al., CVPR 2020
2. **On the Detection of Digital Face Manipulation** - Dang et al., CVPR 2020
3. **Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues** - Qian et al., ECCV 2020
4. **DeeperForensics-1.0** - Jiang et al., CVPR 2020
5. **Celeb-DF** - Li et al., CVPR 2020
6. **Multi-Attentional Deepfake Detection** - Zhao et al., CVPR 2021
7. **Spatial-Phase Shallow Learning** - Liu et al., CVPR 2021
8. **Local Relation Learning for Face Forgery Detection** - Chen et al., AAAI 2021
9. **End-to-End Reconstruction-Classification Learning for Face Forgery Detection** - Cao et al., CVPR 2022
10. **Dual Contrastive Learning for General Face Forgery Detection** - Sun et al., AAAI 2022
11. **UCF: Uncovering Common Features for Generalizable Deepfake Detection** - Yan et al., ICCV 2023
12. **SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes** - Larue et al., ICCV 2023
13. **Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection** - Tsigos et al., 2024
14. **DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection** - Yan et al., NeurIPS 2023
15. **Frequency-Aware Deepfake Detection** - Tan et al., AAAI 2024

The complete 25-paper literature review is documented in the project proposal.

---

# Repository Structure

The repository will evolve as the experimental implementation progresses. The intended organization is:

```text
Robust-Explainable-and-Generalizable-Deepfake-Image-Detection-using-CNNs/
│
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── robustness/
│   └── explainability/
│
├── notebooks/
│   ├── exploratory_analysis/
│   ├── baseline/
│   ├── robustness/
│   ├── explainability/
│   └── generalizability/
│
├── configs/
│
├── checkpoints/
│
├── results/
│   ├── baseline/
│   ├── robustness/
│   ├── explainability/
│   └── generalizability/
│
├── reports/
│
└── requirements.txt
```

> **Note:** The structure above represents the intended organization of the research implementation. Files and directories should only be added to the README as they become available in the repository.

---

# Reproducibility

Reproducibility is a central requirement of this project.

Each experiment should document:

* Dataset version
* Dataset split
* Random seed
* Model architecture
* Preprocessing
* Image resolution
* Training configuration
* Transformation parameters
* Transformation severity
* Evaluation metrics
* Hardware configuration
* Model checkpoint
* Experiment results

The same baseline and evaluation protocol should be reused across the three research verticals whenever possible.

---

# Limitations

The project has several important limitations.

### Dataset Dependence

Results may depend on the datasets and manipulation techniques selected for evaluation.

### Computational Constraints

Large datasets such as DFDC and DeeperForensics-1.0 may require substantial storage and computational resources. The proposal therefore treats them as secondary candidates.

### Explanation Limitations

Grad-CAM provides a spatial visualization of model activations, but it should not be interpreted as a complete or causal representation of the model's reasoning.

### Generalization Scope

Testing a limited number of unseen manipulation techniques or datasets cannot establish universal generalization.

### Architecture Scope

The project deliberately focuses on an existing CNN detector rather than developing a new detection architecture. Consequently, the conclusions primarily describe the behaviour of the selected baseline.

---

# Future Work

Potential extensions include:

* Testing additional CNN architectures
* Cross-dataset evaluation across more benchmarks
* Multi-manipulation attribution
* Frequency-domain analysis
* Stronger quantitative XAI evaluation
* Alternative XAI methods
* Test-time adaptation
* Domain-generalization training
* Manipulation-region supervision
* Ensemble-based detection
* Evaluation on newer generation techniques
* Evaluation under more realistic social-media transformations

These extensions are outside the initial core scope and may be considered after completing the four primary approaches.

---

# Team

| Name                       | Role                 |
| -------------------------- | -------------------- |
| **Dhruv Rakeshbhai Patel** | Project Group Member |
| **Kunta Vikranth Reddy**   | Project Group Member |
| **Saketh Sridharan**       | Project Group Member |
| **Nitish Katteboyina**     | Project Group Member |

**Course:** Advanced Data Mining (CS G520)
**Instructor-in-Charge:** Dr. Hemant Rathore

---

# Dataset Resources

* [FaceForensics++](https://github.com/ondyari/FaceForensics)
* [Celeb-DF](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics)
* [DFDC](https://ai.meta.com/datasets/dfdc/)
* [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0)
* [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)

---

# References

The project proposal contains the complete 25-paper literature review and reference list. Key references include:

1. Li et al. (2020). *Face X-Ray for More General Face Forgery Detection*. CVPR.
2. Dang et al. (2020). *On the Detection of Digital Face Manipulation*. CVPR.
3. Qian et al. (2020). *Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues*. ECCV.
4. Jiang et al. (2020). *DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection*. CVPR.
5. Li et al. (2020). *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*. CVPR.
6. Zhao et al. (2021). *Multi-Attentional Deepfake Detection*. CVPR.
7. Liu et al. (2021). *Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain*. CVPR.
8. Chen et al. (2021). *Local Relation Learning for Face Forgery Detection*. AAAI.
9. Cao et al. (2022). *End-to-End Reconstruction-Classification Learning for Face Forgery Detection*. CVPR.
10. Sun et al. (2022). *Dual Contrastive Learning for General Face Forgery Detection*. AAAI.
11. Yan et al. (2023). *UCF: Uncovering Common Features for Generalizable Deepfake Detection*. ICCV.
12. Larue et al. (2023). *SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes*. ICCV.
13. Yan et al. (2023). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. NeurIPS.
14. Tan et al. (2024). *Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning*. AAAI.
15. Tsigos et al. (2024). *Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection*.

---

# License

This project is released under the [MIT License](LICENSE).

---

## Project Status

**Current Stage:** Proposal / Experimental Design

The current repository represents the research project setup and experimental framework. Model training, robustness experiments, Grad-CAM analysis, and generalization experiments will be added progressively as the project advances.

---

<p align="center">
  <b>Robustness · Explainability · Generalizability</b>
  <br>
  Deepfake Image Detection using CNNs
</p>
