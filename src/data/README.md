# Data Processing Pipeline

This directory contains the complete data preprocessing pipeline for Approach 1.

## Modules

### `inspect_dataset.py`
Verifies the FaceForensics++ dataset structure. Confirms 130 videos per category (650 total) across Original, Deepfakes, Face2Face, FaceSwap, and NeuralTextures.

**Usage:**
```bash
python src/data/inspect_dataset.py
```

### `build_relationship_graph.py`
Parses target_source relationships from manipulated video filenames. Builds a graph where nodes are original video IDs and edges are manipulation relationships. Canonicalizes reverse pairs (e.g., `008_990.mp4` and `990_008.mp4` become one edge). Finds connected components via DFS.

**Usage:**
```bash
python src/data/build_relationship_graph.py
```

### `create_splits.py`
Assigns connected components to train/val/test splits. Prevents leakage by keeping all related videos (same source/target relationship) in one split. Uses seed 42 for reproducibility. Target allocation is 100/20/10 groups, adjusted if the graph structure differs.

**Output:** `data/splits/splits.json`

**Usage:**
```bash
python src/data/create_splits.py
```

### `extract_frames.py`
Extracts every 4th frame from each video (7.5 FPS effective from 30 FPS source). Saves frames to `data/frames/<split>/<label>/<video_id>/frame_*.jpg`. Records metadata for each frame: path, video ID, target ID, source ID, category, label, frame number, timestamp, split.

**Output:** `data/manifests/frame_extraction_metadata.json`

**Usage:**
```bash
python src/data/extract_frames.py
```

### `detect_and_crop_faces.py`
Runs MTCNN (facenet-pytorch) on every selected frame. Tracks faces across frames using IoU-based association. Selects the largest valid face on the first frame, then tracks via IoU (threshold 0.5). Expands bounding boxes by 30%, validates geometry, crops, and overwrites the original frame with the face crop.

Skips frames where detection fails. Excludes videos with fewer than 20 usable frames.

**Output:** `data/manifests/face_processed_metadata.json`

**Usage:**
```bash
python src/data/detect_and_crop_faces.py
```

### `build_manifest.py`
Builds final CSV and JSON manifests from face-processed metadata. The manifest is the authoritative record linking frames to labels, splits, and source videos.

**Output:**
- `data/manifests/manifest.csv`
- `data/manifests/manifest.json`

**Usage:**
```bash
python src/data/build_manifest.py
```

## Full Pipeline

Run all steps in sequence:

```bash
python src/data/inspect_dataset.py
python src/data/build_relationship_graph.py
python src/data/create_splits.py
python src/data/extract_frames.py
python src/data/detect_and_crop_faces.py
python src/data/build_manifest.py
```

Or use the experiment runner:

```bash
python src/run_experiment.py
```

## Key Parameters

From `src/config.py`:

- `FRAME_SAMPLING_INTERVAL = 4` (every 4th frame)
- `MIN_USABLE_FRAMES_PER_VIDEO = 20`
- `BBOX_EXPANSION_MARGIN = 0.3` (30%)
- `IOU_THRESHOLD = 0.5`
- `MIN_FACE_SIZE = 50` pixels

Edit `src/config.py` to change these values.

## Expected Runtime

Processing 650 videos will take several hours depending on hardware:
- Frame extraction: fast (depends on video length)
- Face detection: slow (RetinaFace on ~thousands of frames)

GPU recommended for face detection.
