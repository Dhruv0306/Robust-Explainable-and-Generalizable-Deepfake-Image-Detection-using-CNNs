# Quick Start Guide

This guide gets you running Approach 1 experiments.

## Prerequisites

- Python 3.8+
- CUDA GPU recommended (CPU supported)
- 60-100 GB disk space
- FaceForensics++ C23 dataset downloaded

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Robust-Explainable-and-Generalizable-Deepfake-Image-Detection-using-CNNs

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

1. Download FaceForensics++ C23 videos (see [official repo](https://github.com/ondyari/FaceForensics))
2. Place videos in:
   - `data/datasets/FaceForensics++/original_sequences/youtube/c23/videos/`
   - `data/datasets/FaceForensics++/manipulated_sequences/*/c23/videos/`

Verify structure:
```bash
python src/data/inspect_dataset.py
```

## Run Experiments

### Full Pipeline (Data + Training)

Process 650 videos and train all 9 models:

```bash
python src/run_experiment.py
```

This will take hours to days depending on hardware.

### Skip Data Preprocessing

If data is already processed:

```bash
python src/run_experiment.py --skip-data
```

### Single Model

Train one model with all seeds:

```bash
python src/run_experiment.py --skip-data --model xception
```

Train specific model + seed:

```bash
python src/run_experiment.py --skip-data --model efficientnet_b0 --seed 42
```

## Check Results

Outputs are saved to:
- `data/output/<model>_<pc>_<timestamp>/` - All metrics and visualizations
- `data/checkpoints/<model>_<pc>_<timestamp>/` - Best model weights

Each run directory contains:
- `config.json` - Full configuration
- `training_curves.png` - Loss and accuracy plots
- `test_results.json` - All metrics
- `test_confusion_matrix_*.png` - Confusion matrices
- `test_roc_curve.png` - ROC curve
- `best_checkpoint.pth` - Model weights

## Configuration

Edit `src/config.py` to change:
- Batch sizes
- Face processing parameters
- Training hyperparameters
- Paths

## Troubleshooting

**Out of memory during training:**
- Reduce batch size in `src/config.py`
- Use `--model efficientnet_b0` (smaller than Xception)

**Face detection too slow:**
- Runs faster on GPU
- Adjust `MIN_USABLE_FRAMES_PER_VIDEO` if needed

**Dataset not found:**
- Check paths in `src/config.py`
- Run `python src/data/inspect_dataset.py` to verify

## Next Steps

After collecting baseline results:
1. Analyze per-manipulation performance
2. Compare three architectures
3. Document findings
4. Implement Approach 2 (robustness experiments)

## Documentation

- `src/README.md` - Detailed usage and pipeline description
- `APPROACH_01_SUMMARY.md` - Implementation overview
- `IMPLEMENTATION_CHECKLIST.md` - Verification checklist
- `docs/plans/Approach_01_plan.md` - Full specification

## Support

Check logs in `experiment.log` for detailed execution traces.
