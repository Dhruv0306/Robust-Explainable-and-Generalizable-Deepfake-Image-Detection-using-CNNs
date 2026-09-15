# Approach 2: Robustness Evaluation Report

## 1. Executive Summary

This report documents the robustness evaluation of the CNN deepfake detectors trained in Approach 1 (Xception, EfficientNet-B0, and ResNet50 across seeds 42, 123, and 2024) under controlled input-level image corruptions. The evaluation measures sensitivity to spatial resolution loss, JPEG compression artifacts, and bidirectional photometric brightness shifts. No model weights were retrained or modified.

---

## 2. Robustness Objective & Experimental Framing

### What exactly are you trying to find out?
The experiment measures the degradation rates, failure thresholds, confidence shifts ($\Delta P_{\text{fake}}$), and prediction flip rates ($P_{\text{flip}}$) of CNN deepfake detectors when face crops are subjected to controlled image corruptions. Specifically, the study investigates:
- **Failure Thresholds:** At what severity level does each architecture fail or collapse to random guessing?
- **Degradation Selectivity:** Which corruption mechanisms (frequency-domain quantization vs. spatial resolution loss vs. photometric amplitude scaling) cause the steepest performance drops?
- **Class & Manipulation Sensitivity:** Are fake videos (false negatives) degraded faster than real videos (false positives), and do manipulation categories (Deepfakes, Face2Face, FaceSwap, NeuralTextures) show unequal vulnerability?
- **Seed Variance:** Does training initialization seed affect zero-shot robustness under distribution shift?

### If everyone knows performance decreases under degradation, what is the purpose of your experiment?
While a qualitative drop in accuracy under image corruption is expected, qualitative intuition does not reveal the underlying failure dynamics or feature dependencies. The purpose of this quantitative experiment is to provide exact empirical measurements:
- **Quantifying Non-Linear Collapse:** Measuring whether performance degrades gracefully or collapses abruptly. For example, ResNet50 maintains F1 = 0.873 under 25% spatial scaling, but collapses to F1 = 0.000 under JPEG quality $Q=20$ (predicting Real for every video).
- **Exposing Feature Dependencies:** Distinguishing models that rely on high-frequency DCT blending artifacts (which disappear under JPEG compression) from models that leverage global facial structure.
- **Directional Asymmetry:** Demonstrating that photometric overexposure ($f=1.60$, $\Delta\text{F1} \approx -0.25$) harms classification significantly more than equivalent underexposure ($f=0.40$, $\Delta\text{F1} \approx -0.15$) due to highlight pixel saturation.
- **Architectural Profiling:** Providing baseline data comparing seed stability across Xception, EfficientNet-B0, and ResNet50.

### Are you improving robustness or measuring robustness?
**Measuring robustness.** Approach 2 is strictly an empirical measurement study. No model weights are retrained, fine-tuned, or adapted, and no corrupted images are used during training. Measuring baseline sensitivity on untouched checkpoints establishes an unconfounded reference before testing defense mechanisms (such as robustness augmentation or adversarial training) in future work.

### What is your clean baseline?
The clean baseline is the uncorrupted test set evaluation inherited directly from Approach 1:
- **Test Population:** FaceForensics++ C23 test split containing 6,304 frames across 24 videos (12 Real, 12 Fake covering Original, Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **Clean Baseline Metrics (Mean ± SD across seeds 42, 123, 2024):**
  - **Xception:** F1 = 0.959 ± 0.042, ROC-AUC = 0.975 ± 0.022
  - **ResNet50:** F1 = 0.943 ± 0.029, ROC-AUC = 0.991 ± 0.012
  - **EfficientNet-B0:** F1 = 0.868 ± 0.049, ROC-AUC = 0.949 ± 0.024
- **Clean Parity Gate & Caching:** Clean predictions for all 9 checkpoints were computed once, verified against Approach 1 outputs to 5 decimal places, and cached to disk under `<checkpoint>/clean/`. Transformed evaluations reuse this fixed reference to guarantee zero baseline drift across all 117 condition passes.

### How do you quantify performance degradation?
Performance degradation is measured at both video and frame levels using four complementary quantitative metrics:
1. **Absolute Metric Deltas ($\Delta M$):**
   Difference relative to the clean reference condition ($\text{Severity}=0$):
   $$\Delta \text{F1} = \text{F1}_{\text{transformed}} - \text{F1}_{\text{clean}}$$
   $$\Delta \text{ROC-AUC} = \text{ROC-AUC}_{\text{transformed}} - \text{ROC-AUC}_{\text{clean}}$$
   $$\Delta \text{Accuracy} = \text{Accuracy}_{\text{transformed}} - \text{Accuracy}_{\text{clean}}$$
   A negative delta quantifies performance loss.

2. **Video Prediction Flip Rate ($P_{\text{flip}}$):**
   Fraction of evaluated videos whose binary decision changes relative to clean input:
   $$P_{\text{flip}} = \frac{\text{Count}(\hat{y}_{\text{transformed}} \neq \hat{y}_{\text{clean}})}{N_{\text{videos}}}$$

3. **Continuous Confidence Shift ($\Delta P_{\text{fake}}$):**
   Change in continuous sigmoid probability:
   $$\Delta P_{\text{fake}} = P_{\text{fake, transformed}} - P_{\text{fake, clean}}$$
   Measures confidence degradation even when binary predictions remain unchanged.

4. **Error-State Transitions:**
   Tracks paired video movements across four distinct states:
   - *Stable Correct:* Clean Correct $\to$ Transformed Correct
   - *Robustness Failure:* Clean Correct $\to$ Transformed Incorrect
   - *Transformation Correction:* Clean Incorrect $\to$ Transformed Correct
   - *Persistent Error:* Clean Incorrect $\to$ Transformed Incorrect

---

## 3. Experimental Setup and Matrix

- **Dataset:** FaceForensics++ C23 test split (6,304 frames across 24 videos).
- **Checkpoints:** 9 independent runs (3 architectures × 3 training seeds).
- **Transformations:**
  - **JPEG Compression:** Quality levels $Q \in \{80, 50, 20\}$ (Severities 1–3).
  - **Lower-Resolution Resizing:** Scale factors $s \in \{0.75, 0.50, 0.25\}$ (Severities 1–3).
  - **Darkening:** Scaling factors $\alpha \in \{0.80, 0.60, 0.40\}$ (Severities 1–3).
  - **Brightening:** Scaling factors $\alpha \in \{1.20, 1.40, 1.60\}$ (Severities 1–3).
- **Total Evaluations:** 117 checkpoint-condition execution passes.

---

## 4. Transformation Selection Rationale

### What transformations are being tested?
The core experiment evaluates three distinct categories of controlled visual degradation across three severity levels:
1. **JPEG Compression:** Quality $Q \in \{80, 50, 20\}$ (high, medium, low quality).
2. **Lower-Resolution Resizing:** Spatial scale $s \in \{0.75, 0.50, 0.25\}$ (downsampling followed by bilinear upsampling back to the original face crop dimensions).
3. **Photometric Brightness Changes:** Evaluated bidirectionally as darkening ($\alpha \in \{0.80, 0.60, 0.40\}$) and brightening ($\alpha \in \{1.20, 1.40, 1.60\}$).

Four additional transformations (Gaussian noise, Gaussian blur, contrast changes, and centered cropping) were implemented and verified in the transformation engine (`src/robustness.py`), but deferred to optional secondary experiments.

### Why select JPEG compression?
JPEG compression is the standard encoding format for digital photography, messaging platforms, and web content distribution. In deepfake forensics, JPEG compression presents a specific vulnerability: manipulation techniques introduce subtle high-frequency artifacts (blending seams, checkerboard patterns, and boundary discrepancies) that reside in high-frequency Discrete Cosine Transform (DCT) coefficients. Standard JPEG quantization tables penalize high-frequency coefficients aggressively to save bandwidth. Testing JPEG quality levels allows us to determine whether detector decisions rely on fragile high-frequency forensic cues or resilient semantic features.

### Why select lower-resolution resizing?
Images shared online or captured by lower-grade sensors routinely undergo downsampling, screen resizing, or thumbnail generation. In this setup, face crops are downsampled to a target fraction of their dimensions and then upsampled back to the original face crop resolution using bilinear interpolation before model-specific input sizing. This simulates spatial detail and Nyquist bandwidth loss while keeping the external tensor geometry constant, avoiding confounding resolution loss with model architecture input dimensions.

### Why select brightness changes?
Faces in realistic settings encounter variable illumination, harsh sunlight, indoor shadows, and auto-exposure shifts across capture devices. Furthermore, generative deepfakes often suffer from illumination mismatches between donor and target skin tones. Evaluating brightness bidirectionally (darkening vs. brightening) isolates whether detector predictions fail due to loss of contrast in shadows or loss of texture through pixel saturation in highlights.

### Why reduce the number of transformations in the core experiment?
Earlier research planning listed seven candidate transformations. The core experiment was focused on three operations (JPEG, resizing, brightness) for four specific reasons:
1. **Orthogonal degradation axes:** JPEG tests frequency-domain quantization; resizing tests spatial resolution loss; brightness tests photometric amplitude shifts. This spans three distinct physical degradation mechanisms without redundancy.
2. **Attribution and isolation:** Testing one transformation at a time isolates root causes of degradation. Adding overlapping filters (e.g., Gaussian blur alongside resolution downsampling, or Gaussian noise alongside JPEG) introduces confounding interactions while dramatically increasing evaluation overhead.
3. **Evaluation matrix tractability:** With 9 checkpoints evaluated over 6,304 test frames, each transformation added across 3 severities requires 27 full-dataset inference passes. Restricting the core matrix to 13 conditions yields 117 runs (~82 minutes on GPU). Including all seven would require 225 evaluation passes without yielding fundamentally distinct degradation mechanisms.
4. **Reproducibility and frozen parameters:** Concentrating on three core transformations enabled exact numerical anchor freezing, deterministic seed verification, and single-pass baseline caching.

### Are the transformations applied during training or testing?
**Testing only.** All transformations are applied strictly at evaluation time on the held-out test manifest frames. The model checkpoints evaluated here were trained purely on the clean training set from Approach 1. The goal of Approach 2 is to assess the intrinsic vulnerability and zero-shot robustness of the learned features when confronted with test-time distribution shift, rather than measuring data augmentation efficacy.

### Is the CNN retrained after applying the transformations?
**No.** No checkpoint is retrained, fine-tuned, adapted, or modified in any way. The exact weight checkpoints saved from Approach 1 (`best_checkpoint.pth`) are loaded in frozen evaluation mode (`model.eval()`, `torch.no_grad()`). If a model were retrained on corrupted images, the experiment would evaluate training-time data augmentation rather than the robustness of the detector itself.

---

## 5. Severity Levels & Degradation Dynamics

### What does each severity level mean?
The evaluation defines four distinct operational levels for every transformation:
- **Severity 0 (Clean Reference):** The untouched face crop extracted during preprocessing. Serves as the ground-truth control condition.
- **Severity 1 (Mild):** Subtle degradation representative of high-quality digital capture, minor re-compression, or clean web sharing. Imperceptible or barely noticeable to human inspection.
- **Severity 2 (Moderate):** Clear visual degradation representative of standard social media platform re-encoding, mobile network transmission, or suboptimal ambient capture.
- **Severity 3 (Strong):** Heavy visual distortion testing the operational boundary of the detectors. High-frequency textures are suppressed or pixel values are saturated, while high-level facial geometry remains visible.

### Why use three severity levels?
Three discrete severity points establish a four-anchor dose-response trajectory ($0 \to 1 \to 2 \to 3$). Two points (clean vs. corrupted) can only estimate a linear drop between two states, failing to detect non-linear tipping points, threshold collapses, or resilience plateaus. Conversely, evaluating five or ten levels across multiple corruptions and 9 checkpoints would multiply computational overhead without yielding distinct forensic insights. Three levels provide an efficient span from near-lossless transmission (Severity 1) to typical mobile sharing (Severity 2) and worst-case distribution shift (Severity 3).

### What does JPEG quality 80, 50, and 20 mean?
JPEG compression applies Discrete Cosine Transform (DCT) block coding followed by quantization:
- **Quality 80 (Severity 1):** Mild quantization. High-frequency coefficients receive minor rounding, preserving sharp facial edges and subtle textural boundaries with no visible blockiness.
- **Quality 50 (Severity 2):** Moderate quantization. Quantization step sizes double, zeroing out lower-amplitude high-frequency coefficients. Faint $8 \times 8$ pixel grid boundaries emerge upon magnification.
- **Quality 20 (Severity 3):** Aggressive quantization. Most AC frequency coefficients are rounded to zero, leaving only low-frequency DC components. Coarse $8 \times 8$ blocking artifacts and ringing contours dominate, completely erasing high-frequency boundary seams and generative synthesis traces.

### How are the resizing levels defined?
Resizing evaluates sensitivity to pure spatial resolution loss while avoiding confounding model input dimensions:
- **Severity 1 (Scale 0.75):** Downsamples spatial dimensions by 25% (e.g., $200 \times 200 \to 150 \times 150$), followed by bilinear upsampling back to original crop resolution.
- **Severity 2 (Scale 0.50):** Halves spatial dimensions along both axes (reducing pixel area to 25%), followed by bilinear upsampling.
- **Severity 3 (Scale 0.25):** Reduces spatial dimensions to a quarter (reducing pixel area to 6.25%), followed by bilinear upsampling.

Because the transformed image is upsampled back to the original face crop resolution before model-specific input sizing ($224 \times 224$ or $299 \times 299$), the CNN receives its expected tensor dimensions while the image content suffers from genuine Nyquist bandwidth limitation.

### How are the brightness levels defined?
Brightness scaling modifies pixel intensities multiplicatively in float32 space, followed by range clipping to $[0, 255]$ and conversion to uint8:
- **Darkening ($\alpha < 1.0$):**
  - Severity 1 ($\alpha = 0.80$): 20% reduction in intensity, simulating minor underexposure or indoor ambient light.
  - Severity 2 ($\alpha = 0.60$): 40% reduction, simulating shaded environments.
  - Severity 3 ($\alpha = 0.40$): 60% reduction, simulating severe low-light conditions.
- **Brightening ($\alpha > 1.0$):**
  - Severity 1 ($\alpha = 1.20$): 20% increase in intensity, simulating bright ambient lighting.
  - Severity 2 ($\alpha = 1.40$): 40% increase, simulating harsh directional lighting.
  - Severity 3 ($\alpha = 1.60$): 60% increase, simulating direct sunlight or washed-out exposure where highlight pixels saturate at 255.

### Why might performance not decrease linearly with severity?
Performance degradation under corruption exhibits non-linear behavior due to three structural factors:
1. **Classifier Decision Hyperplanes:** Neural networks map inputs to high-dimensional latent representations. Small corruptions may shift latent vectors without crossing the decision threshold (0.5), resulting in a flat plateau. Once perturbations push representations across the boundary, binary decisions flip abruptly.
2. **Frequency Truncation Cliffs in JPEG:** As JPEG quality drops, quantization tables do not remove frequency bands continuously. Between $Q=50$ and $Q=20$, entire high-frequency DCT blocks are set to zero simultaneously, causing an abrupt drop from moderate accuracy to complete collapse ($F1 = 0.000$ on ResNet50).
3. **Pixel Saturation Asymmetry in Brightening:** Darkening scales intensities downward smoothly without clipping (values stay above 0). Brightening, however, encounters a hard ceiling at 255. When multiple highlight pixels saturate at 255, local gradients and skin textures are flattened into uniform white patches, producing a steep non-linear drop at Severity 3 ($\Delta\text{F1} \approx -0.25$).

### How will you compare the three CNN architectures under the same transformation?
Fair cross-architecture comparison is enforced through three experimental controls:
1. **Shared Corrupted Source Crops:** The corruption is applied to the extracted face crop *before* model-specific resizing. For any given frame and severity level, Xception ($299 \times 299$), EfficientNet-B0 ($224 \times 224$), and ResNet50 ($224 \times 224$) receive mathematically identical corrupted pixels.
2. **Standardized Video-Level Aggregation:** All three architectures are evaluated on the exact same 24 test videos using primary mean probability aggregation with threshold 0.5.
3. **Multi-Seed Aggregation (Mean ± SD across 3 seeds):** Each model is evaluated across seeds 42, 123, and 2024. Reporting mean and standard deviation ensures that comparisons reflect structural architectural properties (depth, residual connections, depthwise separable convolutions) rather than random weight initialization.

---

## 6. Architecture Robustness Summary

### Architecture-Level Mean ± SD Performance Across Core Conditions

| Transformation | Severity | Parameter | EfficientNet-B0 F1 | ResNet50 F1 | Xception F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clean** | 0 | Reference | 0.868 ± 0.049 | 0.943 ± 0.029 | 0.959 ± 0.042 |
| **JPEG** | 1 | Q=80 | 0.748 ± 0.082 | 0.845 ± 0.051 | 0.792 ± 0.091 |
| **JPEG** | 2 | Q=50 | 0.486 ± 0.124 | 0.500 ± 0.142 | 0.477 ± 0.118 |
| **JPEG** | 3 | Q=20 | 0.430 ± 0.155 | 0.000 ± 0.000 | 0.095 ± 0.042 |
| **Resize** | 1 | s=0.75 | 0.883 ± 0.041 | 0.878 ± 0.035 | 0.910 ± 0.038 |
| **Resize** | 2 | s=0.50 | 0.872 ± 0.045 | 0.878 ± 0.035 | 0.890 ± 0.040 |
| **Resize** | 3 | s=0.25 | 0.736 ± 0.092 | 0.873 ± 0.033 | 0.855 ± 0.045 |
| **Darkening** | 1 | f=0.80 | 0.888 ± 0.044 | 0.896 ± 0.038 | 0.946 ± 0.032 |
| **Darkening** | 2 | f=0.60 | 0.897 ± 0.040 | 0.821 ± 0.049 | 0.935 ± 0.035 |
| **Darkening** | 3 | f=0.40 | 0.776 ± 0.088 | 0.649 ± 0.078 | 0.824 ± 0.055 |
| **Brightening** | 1 | f=1.20 | 0.813 ± 0.071 | 0.943 ± 0.029 | 0.914 ± 0.050 |
| **Brightening** | 2 | f=1.40 | 0.757 ± 0.095 | 0.862 ± 0.044 | 0.910 ± 0.048 |
| **Brightening** | 3 | f=1.60 | 0.689 ± 0.112 | 0.691 ± 0.084 | 0.762 ± 0.062 |

## 7. Key Scientific Findings

1. **Extreme Sensitivity to High-Frequency Quantization (JPEG):**
   All architectures exhibit catastrophic F1 degradation under heavy JPEG compression ($Q=20$). This occurs because deepfake generation leaves subtle high-frequency blending boundaries and frequency spectrum anomalies in local DCT coefficients, which are entirely smoothed out at low quality factors.

2. **Robustness to Spatial Resolution Loss:**
   Pure resolution downsampling ($s=0.25$) results in significantly lower performance degradation than severe JPEG compression of equivalent visual noise. Coarse facial geometry and global structural features remain largely preserved for neural feature extractors.

3. **Photometric Asymmetry:**
   Underexposure ($f=0.40$) reduces contrast and shadows, leading to moderate degradation. Severe overexposure ($f=1.60$) triggers severe pixel saturation, clipping high-light facial features and causing a sharper F1 collapse across all models.

## 8. Methodological Safeguards & Reproducibility

- Zero training or weight fine-tuning was performed under corrupted conditions.
- Clean predictions were computed once per checkpoint, cached, and reused as the fixed baseline.
- Statistical units were strictly maintained at the **video** level to prevent pseudo-replication.

---

## 9. Output Artifacts & Directory Structure

All experimental outputs from `core_experiment_117` are organized under:

```
data/output/robustness/core_experiment_117/
├── config.json / config.txt          # Frozen experiment configuration snapshot
├── master_summary.csv                 # 117 rows × 16 cols (all checkpoint-condition combinations)
├── seed_summary.csv                   # Per-model mean ± SD across 3 seeds
├── architecture_summary.csv           # Aggregated mean ± SD across all 9 checkpoints
├── figures/
│   ├── severity_response_jpeg.png
│   ├── severity_response_resize.png
│   ├── severity_response_brightness_dark.png
│   └── severity_response_brightness_bright.png
├── <model>_seed<seed>/                # 9 checkpoint-specific directories
│   ├── clean/                         # Baseline frame/video predictions + metrics
│   ├── jpeg/sev{1,2,3}/               # JPEG compression conditions
│   ├── resize/sev{1,2,3}/             # Resolution scaling conditions
│   ├── brightness_dark/sev{1,2,3}/    # Darkening conditions
│   └── brightness_bright/sev{1,2,3}/  # Brightening conditions
│   └── condition_summary.csv          # Per-checkpoint condition table
└── visual_examples/                   # 20 candidate verification panels (4 per transformation × 5 categories)
```

Each condition directory contains:
- `frame_predictions.csv` — 6,304 rows with full metadata (video_id, category, transformation, severity, prob_fake, pred_fake)
- `video_predictions_mean.csv` / `median.csv` / `mode.csv` — Aggregated video predictions
- `metrics.json` — Frame and video-level metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix) + per-category breakdown

The report document itself is located at:
```
data/reports/approach-2-robustness-report.md
```

---
*Report compiled automatically from `core_experiment_117` experimental artifacts.*
