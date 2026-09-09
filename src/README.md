# Approach 1: CNN Baseline Implementation

**Binary deepfake image classification using CNNs trained on face-cropped frames from FaceForensics++.**

## Overview

This implementation follows the complete Approach 1 protocol from `docs/plans/Approach_01_plan.md`:

- **Task**: Binary Real vs Fake classification
- **Dataset**: FaceForensics++ C23 (130 videos × 5 categories = 650 videos)
- **Models**: Xception, EfficientNet-B0, ResNet50 (ImageNet pretrained via timm)
- **Experiments**: 3 models × 3 seeds = 9 runs
- **Evaluation**: Video-level aggregation (mean/median/mode) with per-manipulation breakdown

## Project Structure

```
src/
├── config.py                    # All constants, paths, hyperparameters
├── utils.py                     # Seed, device, logging utilities
├── models.py                    # CNN definitions via timm
├── dataset.py                   # PyTorch Dataset + augmentation
├── train.py                     # Training loop
├── evaluate.py                  # Evaluation + video aggregation
├── run_experiment.py            # Full pipeline CLI
└── data/
    ├── inspect_dataset.py       # Dataset inspection
    ├── build_relationship_graph.py  # Parse target/source relationships
    ├── create_splits.py         # Leakage-safe splitting
    ├── extract_frames.py        # Every 4th frame extraction
    ├── detect_and_crop_faces.py # RetinaFace + IoU tracking
    └── build_manifest.py        # CSV/JSON manifest generation
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use existing venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Usage

### Full Pipeline (Data + Training)

Run the complete experiment matrix (3 models × 3 seeds):

```bash
python src/run_experiment.py
```

### Skip Data Preprocessing (if already done)

```bash
python src/run_experiment.py --skip-data
```

### Run Single Model

```bash
# All seeds for one model
python src/run_experiment.py --skip-data --model xception

# Specific model + seed
python src/run_experiment.py --skip-data --model efficientnet_b0 --seed 42
```

### Individual Pipeline Steps

```bash
# 1. Inspect dataset
python src/data/inspect_dataset.py

# 2. Build relationship graph
python src/data/build_relationship_graph.py

# 3. Create splits
python src/data/create_splits.py

# 4. Extract frames
python src/data/extract_frames.py

# 5. Detect and crop faces
python src/data/detect_and_crop_faces.py

# 6. Build manifest
python src/data/build_manifest.py

# 7. Train single model
python src/train.py

# 8. Evaluate
python src/evaluate.py
```

## Data Pipeline

1. **Inspect**: Verify 130 videos per category
2. **Graph**: Parse target_source relationships, canonicalize reverse pairs
3. **Split**: Connected components → train/val/test (no leakage)
4. **Frames**: Extract every 4th frame (7.5 FPS effective)
5. **Faces**: RetinaFace detection + IoU tracking + bbox expansion + crop
6. **Manifest**: CSV + JSON with frame metadata

**Output**:
- `data/frames/<split>/<label>/<video_id>/frame_*.jpg`
- `data/manifests/manifest.csv`
- `data/manifests/manifest.json`
- `data/splits/splits.json`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (patience=2) |
| Early stopping | patience=5 |
| Max epochs | 30 |
| Loss | BCEWithLogitsLoss + class weights |
| Augmentation | HFlip (p=0.5) + GaussianBlur (p=0.1) |
| AMP | Enabled on CUDA |
| Batch size | Auto-determined |

## Model Outputs

Each run creates:

```
data/output/<model>_<pc>_<timestamp>/
├── config.json                     # Machine-readable config
├── config.txt                      # Human-readable config
├── history.json                    # Training curves
├── best_checkpoint.pth             # Best model
├── test_frame_predictions.csv      # Frame-level predictions
├── test_video_predictions_mean.csv # Video-level (mean)
├── test_video_predictions_median.csv
├── test_video_predictions_mode.csv
└── test_results.json               # All metrics

data/checkpoints/<model>_<pc>_<timestamp>/
├── best_checkpoint.pth
├── config.json
└── config.txt
```

## Evaluation

**Primary**: Video-level mean aggregation  
**Secondary**: Median, mode aggregation  
**Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix  
**Breakdown**: Per-manipulation (Deepfakes, Face2Face, FaceSwap, NeuralTextures)

## Hardware Support

- **CPU**: Supported with conservative batch size
- **GPU**: Automatic batch sizing + AMP
- **Platforms**: Windows, Ubuntu, macOS

## Reproducibility

- Fixed seeds: 42, 123, 2024
- Deterministic splits (seed=42)
- Identical preprocessing across all runs
- Architecture-specific differences limited to: CNN arch, input size, normalization

## Implementation Notes

- **Ponytail mode**: Minimal dependencies, stdlib-first, no abstractions
- **RetinaFace**: GPU/CPU compatible, IoU-based tracking (no SORT/DeepSORT)
- **Face tracking**: Every 4th frame detection, IoU association with prev frame
- **Bbox expansion**: 30% margin (configurable in `config.py`)
- **Video exclusion**: <20 usable frames → skip video
- **Class weights**: Computed from training set only

## Configuration

Edit `src/config.py` to adjust:
- Paths
- Frame sampling interval
- Face processing parameters (IoU threshold, bbox margin, min face size)
- Training hyperparameters
- Batch sizes

## Next Steps

After Approach 1 completes:
- Push results to GitHub (checkpoints may be too large for git)
- Compare 3 architectures across 3 seeds
- Analyze per-manipulation performance
- Document baseline for Approach 2 (robustness experiments)

---

**Status**: Implementation complete, ready to run experiments.
