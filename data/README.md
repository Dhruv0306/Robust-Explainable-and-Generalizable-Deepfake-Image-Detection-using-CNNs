# Data Directory

This directory stores all datasets, processed outputs, and experiment results.

## Structure

```
data/
├── datasets/               # Raw video datasets
│   └── FaceForensics++/
│       ├── original_sequences/youtube/c23/videos/  (130 videos)
│       └── manipulated_sequences/
│           ├── Deepfakes/c23/videos/      (130 videos)
│           ├── Face2Face/c23/videos/      (130 videos)
│           ├── FaceSwap/c23/videos/       (130 videos)
│           └── NeuralTextures/c23/videos/ (130 videos)
│
├── splits/                 # Train/val/test assignments
│   └── splits.json         # Video ID → split mapping
│
├── frames/                 # Extracted face crops
│   ├── train/
│   │   ├── real/<video_id>/frame_*.jpg
│   │   └── fake/<video_id>/frame_*.jpg
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── manifests/              # Frame-to-label mappings
│   ├── manifest.csv        # Primary manifest (used by training)
│   ├── manifest.json
│   ├── frame_extraction_metadata.json
│   └── face_processed_metadata.json
│
├── output/                 # Training run outputs
│   └── <model>_<pc>_<timestamp>/
│       ├── config.json
│       ├── config.txt
│       ├── history.json
│       ├── training_curves.png
│       ├── best_checkpoint.pth
│       ├── test_frame_predictions.csv
│       ├── test_video_predictions_*.csv
│       ├── test_results.json
│       ├── test_confusion_matrix_*.png
│       └── test_roc_curve.png
│
└── checkpoints/            # Best model checkpoints
    └── <model>_<pc>_<timestamp>/
        ├── best_checkpoint.pth
        ├── config.json
        └── config.txt
```

## Dataset Download

Download FaceForensics++ C23 videos using the official download script:

```bash
python data/scripts/faceforensics_download_v4.py
```

See the [official repository](https://github.com/ondyari/FaceForensics) for authentication and download instructions.

## Storage Requirements

- Raw videos (650): ~30-50 GB
- Extracted frames: ~20-40 GB (depends on video length and face detection success rate)
- Model checkpoints: ~100-200 MB per run (9 runs = ~1-2 GB)
- Predictions and metrics: <1 GB

Total: ~60-100 GB

## Gitignore

All dataset files, frames, checkpoints, and outputs are excluded from git via `.gitignore`. Only manifests and splits (small JSON/CSV files) should be committed after verification.

## Notes

Frame extraction and face detection will create thousands of JPEG files. The pipeline logs which videos are excluded due to insufficient usable frames (<20).

After processing completes, verify:
- `data/manifests/manifest.csv` contains frame records
- `data/splits/splits.json` contains train/val/test video IDs
- `data/frames/` contains subdirectories for each split
