# Approach 1 plan

---

## 1. Objective

Approach 1 establishes the primary baseline for binary deepfake image classification using CNNs trained on face-cropped frames extracted from FaceForensics++ videos.

The classifier will distinguish:

- Real: Original
- Fake: DeepFake, Face2Face, FaceSwap, NeuralTextures

The approach compares Xception, EfficientNet-B0, and ResNet50 under the same dataset construction, leakage-safe split, frame sampling, face-processing pipeline, augmentation policy, optimization strategy, evaluation protocol, and random seeds. Architecture-specific differences are limited to the CNN architecture, required input resolution, and corresponding ImageNet normalization.

---

## 2. Dataset

The primary dataset is FaceForensics++ using the C23 compression setting.

The working subset contains 130 videos from each category:

| Category       | Label |  Videos |
| -------------- | ----: | ------: |
| Original       |  Real |     130 |
| DeepFake       |  Fake |     130 |
| Face2Face      |  Fake |     130 |
| FaceSwap       |  Fake |     130 |
| NeuralTextures |  Fake |     130 |
| **Total**      |       | **650** |

Manipulated filenames follow:

`targetID_sourceID.mp4`

where `targetID` identifies the original video on which the manipulation is applied and `sourceID` identifies the original video from which the source face is taken.

Reverse relationships such as `008_990.mp4` and `990_008.mp4` are treated as the same unordered relationship during split construction.

---

## 3. Leakage-safe splitting

Individual manipulated videos will not be randomly assigned to train, validation, and test sets.

A relationship graph will be constructed:

- Each original video ID is a node.
- Each target-source relationship is an edge.
- Reverse pairs are canonicalized as one unordered pair.
- Connected components are treated as indivisible split groups.

The planned allocation is 100 groups for training, 20 for validation, and 10 for testing, but only if the verified graph supports exactly 130 groups. The graph structure takes precedence over forcing these counts.

The split must verify that:

1. No original video ID occurs in more than one split.
2. No manipulated video occurs in more than one split.
3. Reverse target-source relationships remain within one split.
4. Related original and manipulated videos do not cross split boundaries.
5. All five categories are represented.
6. Assignment is deterministic through a fixed seed.

If an individual video later fails preprocessing, only that video is excluded.

---

## 4. Frame extraction

The source videos are treated as 30 FPS.

Every fourth frame will be retained:

`0, 4, 8, 12, ...`

This gives an effective sampling rate of 7.5 FPS.

The same deterministic rule is applied to every video.

For each retained and successfully processed frame, the pipeline records:

- Frame path
- Video ID
- Target ID
- Source ID
- Category
- Binary label
- Original frame number
- Timestamp
- Split

---

## 5. Face processing

Face cropping will be applied before CNN training. Dlib will not be used.

The pipeline will include:

1. Face detection.
2. Face tracking across selected frames.
3. Bounding-box processing.
4. Face crop generation.
5. Crop validation.
6. Recording of detection and tracking failures.

The exact detector, tracker, initialization frequency, bounding-box expansion, multiple-face handling, and failure policy will be finalized before implementation.

If a selected frame does not produce a usable face crop, that frame is skipped and the failure is recorded. The entire video is not automatically discarded.

A video must contain at least 20 successfully cropped frames. Videos with fewer than 20 usable frames are excluded individually and the exclusion reason is recorded.

---

## 6. Processed dataset organization

The processed dataset will use a structured directory layout:

```text
frames/
├── train/
│   ├── real/
│   │   └── 001/
│   │       ├── frame_000000.jpg
│   │       └── frame_000004.jpg
│   └── fake/
│       └── Deepfakes_001_042/
│           ├── frame_000000.jpg
│           └── frame_000004.jpg
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

The exact directory naming convention will be kept consistent with the final dataset parser.

---

## 7. Dataset manifests

Both CSV and JSON manifests will be generated.

Preferred fields:

```text
frame_path
video_id
target_id
source_id
category
label
original_frame_number
timestamp
split
```

For Original videos, `source_id` is null and `target_id` may correspond to the original video ID.

The manifest is the authoritative mapping between processed frames, source videos, labels, and splits. It also supports video-level aggregation during evaluation.

---

## 8. Class balancing

All successfully extracted training frames will be retained. Fake frames will not be downsampled.

Class weights will be calculated from training frames only, after extraction.

The primary loss is:

`BCEWithLogitsLoss`

with class weighting to account for class imbalance.

Validation and test data will not be used when calculating training class weights.

---

## 9. CNN architectures

Three architectures will be evaluated:

| Model           | Pretraining | Input size | Output    |
| --------------- | ----------- | ---------: | --------- |
| Xception        | ImageNet    |  299 × 299 | One logit |
| EfficientNet-B0 | ImageNet    |  224 × 224 | One logit |
| ResNet50        | ImageNet    |  224 × 224 | One logit |

FaceForensics-specific pretrained Xception weights will not be used for the primary comparison because this would introduce architecture-specific pretraining advantages.

---

## 10. Image preprocessing

Face crops are resized to the required input resolution for each architecture and normalized according to the corresponding ImageNet preprocessing.

Apart from required input size and normalization, preprocessing remains consistent across models.

---

## 11. Training augmentation

Approach 1 uses:

- Random horizontal flip
- Controlled Gaussian blur

The following robustness transformations are excluded from Approach 1 training augmentation:

- JPEG compression
- Lower-resolution resizing
- Brightness changes

These transformations are reserved for later robustness experiments.

---

## 12. Optimization and training

Optimizer:

- AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-4`

Scheduler:

- ReduceLROnPlateau
- Monitor: validation loss

Training limits:

- Maximum epochs: 30
- Early stopping patience: 5
- Primary checkpoint criterion: lowest validation loss

Batch size will be selected safely according to available hardware and remain configurable.

---

## 13. Hardware and platform support

The implementation must support CPU and GPU execution and remain portable across:

- Windows
- Ubuntu
- macOS

Paths will use platform-independent path handling. Device selection and hardware-dependent settings such as batch size will not be hard-coded for one machine.

---

## 14. Checkpoint selection

The primary checkpoint is the epoch with the lowest validation loss.

The following validation metrics are also recorded:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

F1 and ROC-AUC are secondary model-selection diagnostics. Precision and recall are retained as diagnostics but neither is used alone to select the checkpoint.

The selected checkpoint is evaluated on the test set.

---

## 15. Reproducibility

Each CNN will be trained with three random seeds:

```text
42
123
2024
```

The following remain identical across architectures and seeds wherever applicable:

- Dataset split
- Video membership
- Frame sampling
- Face-processing configuration
- Labels
- Training augmentation
- Validation data
- Test data
- Evaluation procedure

Architecture, input resolution, and corresponding normalization are the intended model-specific differences.

Every run records its seed and configuration.

---

## 16. Frame-level inference

Each processed face crop produces one logit.

The sigmoid function converts the logit into a fake probability:

`p(fake) = sigmoid(logit)`

Frame-level predictions are retained for supplementary analysis and video-level aggregation.

---

## 17. Video-level aggregation

The primary evaluation unit is the video, not the frame.

For each video, frame-level fake probabilities are aggregated using:

1. Mean probability
2. Median probability
3. Mode of binary frame predictions

Mean probability is the primary aggregation method.

Median probability is a secondary aggregation method.

For mode aggregation, each frame probability is thresholded at 0.5:

- `p(fake) >= 0.5` → Fake
- `p(fake) < 0.5` → Real

The mode of these binary predictions becomes the video-level prediction.

---

## 18. Evaluation metrics

Primary evaluation is performed at the video level.

Reported metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Frame-level metrics may be reported as supplementary diagnostics, but they do not replace video-level evaluation.

ROC-AUC uses the continuous video-level fake probability rather than the thresholded class label.

---

## 19. Per-manipulation evaluation

The classifier remains binary Real/Fake, but fake-video performance is also reported separately for:

- DeepFake
- Face2Face
- FaceSwap
- NeuralTextures

Original videos remain the Real class.

---

## 20. Experiment matrix

The primary experiment contains:

`3 CNN architectures × 3 random seeds = 9 training runs`

Models:

- Xception
- EfficientNet-B0
- ResNet50

Seeds:

- 42
- 123
- 2024

Each run uses the same processed dataset and split configuration.

---

## 21. Experiment outputs

All Approach 1 run outputs will be stored under `data/output/`.

Each model run will create its own directory using the format:

```text
<model_name>_<pc_name>_<date_time_of_start>
```

where:

* `<model_name>` identifies the CNN architecture.
* `<pc_name>` identifies the computer or execution environment used for the run.
* `<date_time_of_start>` records the date and time at which the run started.

For example:

```text
data/output/
├── Xception_PC1_2026-09-05_16-14-00/
├── EfficientNet-B0_PC1_2026-09-05_18-30-00/
└── ResNet50_PC1_2026-09-06_09-15-00/
```

The run directory will contain all outputs associated with that specific training run. It must also contain a `.txt` or `.json` configuration file containing all configuration parameters used for the run.

The configuration record must contain all information required to reproduce and identify the run, including the model, seed, dataset and split configuration, preprocessing, augmentation, optimizer, scheduler, training settings, device, batch size, and relevant runtime settings.

The final selected checkpoint must be saved in both locations:

* The corresponding model run directory under `data/output/`.
* The corresponding model run directory under `data/checkpoints/`.

Checkpoint storage will use the same run-directory naming convention:

```text
data/checkpoints/
├── Xception_PC1_2026-09-05_16-14-00/
├── EfficientNet-B0_PC1_2026-09-05_18-30-00/
└── ResNet50_PC1_2026-09-06_09-15-00/
```

The run directory name must be identical in `data/output/` and `data/checkpoints/` so that the two locations can be directly associated with the same training run.

The final model outputs and required experiment artifacts will be pushed to the project's GitHub repository after the run is completed and the outputs have been verified, subject to repository storage and version-control constraints.

Expected outputs include:

* Training and validation loss histories
* Training and validation metrics
* Best epoch
* Selected checkpoint
* Test predictions
* Frame-level predictions
* Video-level predictions
* Confusion matrices
* ROC curves
* Metric summaries
* Per-manipulation results
* Run configuration
* Random seed

---

## 22. Implementation structure

A proposed implementation structure is:

```text
src/
├── data/
│   ├── inspect_dataset.py
│   ├── build_relationship_graph.py
│   ├── create_splits.py
│   ├── extract_frames.py
│   ├── detect_and_track_faces.py
│   └── build_manifest.py
├── models/
│   ├── xception.py
│   ├── efficientnet.py
│   └── resnet.py
├── training/
│   ├── train.py
│   ├── losses.py
│   ├── scheduler.py
│   └── checkpoint.py
├── evaluation/
│   ├── frame_metrics.py
│   ├── video_aggregation.py
│   └── video_metrics.py
└── utils/
    ├── config.py
    ├── reproducibility.py
    ├── device.py
    └── logging.py
```

This structure can be adapted to the existing repository without changing the experimental protocol.

---

## 23. End-to-end pipeline

```text
FaceForensics++ C23 videos
        ↓
Dataset inspection
        ↓
Parse target/source relationships
        ↓
Canonicalize reverse relationships
        ↓
Build relationship graph
        ↓
Find connected components
        ↓
Create leakage-safe train/validation/test splits
        ↓
Extract every 4th frame
        ↓
Detect faces
        ↓
Track faces across selected frames
        ↓
Crop and validate faces
        ↓
Skip and record failed frames
        ↓
Exclude videos with <20 usable frames
        ↓
Build CSV + JSON manifests
        ↓
Apply model-specific resizing/normalization
        ↓
Train Xception / EfficientNet-B0 / ResNet50
        ↓
3 random seeds per model
        ↓
Select checkpoint using lowest validation loss
        ↓
Run test inference
        ↓
Aggregate frame predictions to video predictions
        ↓
Compute video-level metrics
        ↓
Report per-manipulation performance
```

---

## 24. Fixed configuration

| Component                                | Configuration                                              |
| ---------------------------------------- | ---------------------------------------------------------- |
| Task                                     | Binary Real vs Fake                                        |
| Dataset                                  | FaceForensics++                                            |
| Compression                              | C23                                                        |
| Categories                               | Original, DeepFake, Face2Face, FaceSwap, NeuralTextures    |
| Videos/category                          | 130                                                        |
| Total videos                             | 650                                                        |
| Split principle                          | Source/target relationship graph                           |
| Reverse pairs                            | Canonicalized as unordered relationships                   |
| Planned split                            | 100 / 20 / 10 groups, subject to graph verification        |
| Frame rate assumption                    | 30 FPS                                                     |
| Frame sampling                           | Every 4th frame                                            |
| Effective sampling rate                  | 7.5 FPS                                                    |
| Face processing                          | Face cropping + tracking                                   |
| Face detector                            | To be finalized                                            |
| Tracker                                  | To be finalized                                            |
| Failed detection                         | Skip frame + record failure                                |
| Minimum usable frames                    | 20 per video                                               |
| Training frames                          | All successfully extracted frames                          |
| Class balancing                          | Class-weighted loss                                        |
| Class weights                            | Training frames only                                       |
| Loss                                     | BCEWithLogitsLoss                                          |
| Models                                   | Xception, EfficientNet-B0, ResNet50                        |
| Pretraining                              | ImageNet                                                   |
| Xception input                           | 299 × 299                                                  |
| EfficientNet-B0 input                    | 224 × 224                                                  |
| ResNet50 input                           | 224 × 224                                                  |
| Training augmentation                    | Random horizontal flip + controlled Gaussian blur          |
| Robustness transformations in Approach 1 | None                                                       |
| Optimizer                                | AdamW                                                      |
| Learning rate                            | 1e-4                                                       |
| Weight decay                             | 1e-4                                                       |
| Scheduler                                | ReduceLROnPlateau                                          |
| Maximum epochs                           | 30                                                         |
| Early stopping patience                  | 5                                                          |
| Primary checkpoint criterion             | Lowest validation loss                                     |
| Secondary diagnostics                    | F1, ROC-AUC, precision, recall                             |
| Seeds                                    | 42, 123, 2024                                              |
| Runs                                     | 9                                                          |
| Primary evaluation unit                  | Video                                                      |
| Aggregation                              | Mean, median, binary mode                                  |
| Primary aggregation                      | Mean probability                                           |
| Metrics                                  | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Per-manipulation analysis                | Yes                                                        |
| Manifest formats                         | CSV + JSON                                                 |
| Hardware                                 | CPU + GPU                                                  |
| Platforms                                | Windows + Ubuntu + macOS                                   |
| Minimum video-level criterion            | 20 usable cropped frames                                   |
| Output root                              | `data/output/`                                             |
| Checkpoint root                          | `data/checkpoints/`                                        |
| Run directory                            | `<model_name>_<pc_name>_<date_time_of_start>`              |
| Run configuration                        | `.txt` or `.json` inside the output run directory          |
| Final checkpoint                         | Saved in both output and checkpoint run directories        |
| GitHub                                   | Final model outputs pushed after verification              |

---

## 25. Items to finalize before implementation

The overall Approach 1 protocol is fixed. The following implementation details remain to be finalized before the preprocessing pipeline is written:

1. Face detector.
2. Face tracker.
3. Detector initialization frequency.
4. Bounding-box expansion policy.
5. Handling of multiple faces.
6. Tracker failure and re-detection policy.
7. Face crop validation rules.
8. Exact Gaussian blur probability and parameter range.
9. Exact horizontal-flip probability.
10. Automatic batch-size policy.
11. Exact configuration-file format and integration with the existing repository.

---
