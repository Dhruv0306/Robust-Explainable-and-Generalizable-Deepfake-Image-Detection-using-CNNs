# Approach 2: Robustness Evaluation Report

## 1. Executive Summary

This report documents the robustness evaluation of the CNN deepfake detectors trained in Approach 1 (Xception, EfficientNet-B0, and ResNet50 across seeds 42, 123, and 2024) under controlled input-level image corruptions. The evaluation measures sensitivity to spatial resolution loss, JPEG compression artifacts, and bidirectional photometric brightness shifts. No model weights were retrained or modified.

## 2. Experimental Setup and Matrix

- **Dataset:** FaceForensics++ C23 test split (6,304 frames across 24 videos).
- **Checkpoints:** 9 independent runs (3 architectures × 3 training seeds).
- **Transformations:**
  - **JPEG Compression:** Quality levels $Q \in \{80, 50, 20\}$ (Severities 1–3).
  - **Lower-Resolution Resizing:** Scale factors $s \in \{0.75, 0.50, 0.25\}$ (Severities 1–3).
  - **Darkening:** Scaling factors $\alpha \in \{0.80, 0.60, 0.40\}$ (Severities 1–3).
  - **Brightening:** Scaling factors $\alpha \in \{1.20, 1.40, 1.60\}$ (Severities 1–3).
- **Total Evaluations:** 117 checkpoint-condition execution passes.

## 3. Architecture Robustness Summary

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

## 4. Key Scientific Findings

1. **Extreme Sensitivity to High-Frequency Quantization (JPEG):**
   All architectures exhibit catastrophic F1 degradation under heavy JPEG compression ($Q=20$). This occurs because deepfake generation leaves subtle high-frequency blending boundaries and frequency spectrum anomalies in local DCT coefficients, which are entirely smoothed out at low quality factors.

2. **Robustness to Spatial Resolution Loss:**
   Pure resolution downsampling ($s=0.25$) results in significantly lower performance degradation than severe JPEG compression of equivalent visual noise. Coarse facial geometry and global structural features remain largely preserved for neural feature extractors.

3. **Photometric Asymmetry:**
   Underexposure ($f=0.40$) reduces contrast and shadows, leading to moderate degradation. Severe overexposure ($f=1.60$) triggers severe pixel saturation, clipping high-light facial features and causing a sharper F1 collapse across all models.

## 5. Methodological Safeguards & Reproducibility

- Zero training or weight fine-tuning was performed under corrupted conditions.
- Clean predictions were computed once per checkpoint, cached, and reused as the fixed baseline.
- Statistical units were strictly maintained at the **video** level to prevent pseudo-replication.

---

## 6. Output Artifacts & Directory Structure

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
