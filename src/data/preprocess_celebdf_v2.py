#!/usr/bin/env python3
"""
Preprocess Celeb-DF v2 for the project's cross-dataset evaluation.

The output mirrors the relevant FaceForensics++ preprocessing conventions:
- only the official Celeb-DF v2 test protocol is used by default
- every 4th decoded frame is sampled
- MTCNN detects faces
- the selected face is tracked with IoU across frames
- the bounding box is expanded by 30%
- videos with fewer than 20 usable face frames are excluded
- output frames are stored as:
    <out>/frames/test/<label>/<video_id>/frame_XXXXXX.jpg
- a manifest compatible with the project's FF++ manifest schema is written

Default evaluation set:
    List_of_testing_videos.txt, i.e. the official 518-video Celeb-DF v2 test set.

Subset mode:
    --N selects exactly N videos, where possible, using deterministic
    stratified sampling from the official test list. The real/fake ratio
    of the official test list is preserved as closely as possible.

This script does NOT fine-tune, split, or augment Celeb-DF. It prepares the
frozen cross-dataset evaluation data only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: facenet-pytorch. "
        "Install with: pip install facenet-pytorch"
    ) from exc


FRAME_SAMPLING_INTERVAL = 4
FPS_ASSUMPTION = 30.0
BBOX_EXPANSION_MARGIN = 0.30
IOU_THRESHOLD = 0.50
MIN_FACE_SIZE = 50
MIN_USABLE_FRAMES_PER_VIDEO = 20

# Reproducible subset selection.
SUBSET_SEED = 42

# The official repository uses these directory names.
REAL_DIRS = ("Celeb-real", "YouTube-real")
FAKE_DIR = "Celeb-synthesis"

MANIFEST_COLUMNS = [
    "frame_path",
    "video_id",
    "target_id",
    "source_id",
    "category",
    "label",
    "original_frame_number",
    "timestamp",
    "split",
    "face_bbox",
    "face_detected",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def find_dataset_root(path: Path) -> Path:
    """Resolve a user-provided Celeb-DF root and validate its structure."""
    path = path.expanduser().resolve()

    candidates = [path]
    if path.is_dir():
        candidates.extend(
            [
                path / "Celeb-DF",
                path / "Celeb-DF-v2",
                path / "Celeb-DF-V2",
                path / "Celeb_DF",
            ]
        )

    for candidate in candidates:
        if not candidate.is_dir():
            continue

        has_real = any((candidate / d).is_dir() for d in REAL_DIRS)
        has_fake = (candidate / FAKE_DIR).is_dir()
        test_list = candidate / "List_of_testing_videos.txt"

        if has_real and has_fake and test_list.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find a valid Celeb-DF v2 root. Expected:\n"
        "  Celeb-real/\n"
        "  YouTube-real/\n"
        "  Celeb-synthesis/\n"
        "  List_of_testing_videos.txt"
    )


def normalize_name(value: str) -> str:
    return Path(value.strip().strip('"').strip("'")).name


def read_test_list(root: Path) -> List[str]:
    """
    Read the official test list.

    The file is treated as authoritative. Blank lines and comments are ignored.
    Both bare filenames and relative paths are accepted.
    """
    path = root / "List_of_testing_videos.txt"
    names: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name = normalize_name(line)
            if name:
                names.append(name)

    # Preserve first occurrence while removing accidental duplicates.
    seen = set()
    unique = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    if not unique:
        raise ValueError(f"No video names found in {path}")

    return unique


def build_video_index(root: Path) -> Dict[str, Tuple[Path, str, str]]:
    """
    Map filename -> (path, label, category).

    Category values are deliberately normalized to the project's binary
    cross-dataset convention:
        Original -> real
        Celeb-synthesis -> fake
    YouTube-real is also real.
    """
    index: Dict[str, Tuple[Path, str, str]] = {}

    for directory in REAL_DIRS:
        folder = root / directory
        for p in sorted(folder.glob("*.mp4")):
            index[p.name] = (p, "real", "Original")

    fake_folder = root / FAKE_DIR
    for p in sorted(fake_folder.glob("*.mp4")):
        index[p.name] = (p, "fake", "Deepfakes")

    return index


def select_videos(
    test_names: List[str],
    video_index: Dict[str, Tuple[Path, str, str]],
    n: Optional[int],
) -> List[Tuple[Path, str, str]]:
    """
    Select the official test set or a deterministic stratified subset.

    If --N is supplied, N means total videos. The subset preserves the
    real/fake ratio of the official test list as closely as possible.
    """
    available = []
    missing = []

    for name in test_names:
        item = video_index.get(name)
        if item is None:
            missing.append(name)
        else:
            available.append((name, *item))

    if missing:
        logging.warning(
            "%d test-list entries were not found in the downloaded dataset.",
            len(missing),
        )
        for name in missing[:10]:
            logging.warning("Missing: %s", name)
        if len(missing) > 10:
            logging.warning("... and %d more.", len(missing) - 10)

    if not available:
        raise RuntimeError("None of the official test-list videos were found.")

    if n is None:
        selected = available
    else:
        if n <= 0:
            raise ValueError("--N must be a positive integer.")
        if n > len(available):
            raise ValueError(
                f"--N={n} exceeds the {len(available)} available official test videos."
            )

        by_label = defaultdict(list)
        for item in available:
            by_label[item[2]].append(item)

        rng = random.Random(SUBSET_SEED)

        # Preserve the official real/fake proportions.
        counts = {
            label: len(items) for label, items in by_label.items()
        }
        labels = sorted(counts)

        raw_targets = {
            label: n * counts[label] / len(available) for label in labels
        }
        target_counts = {
            label: int(math.floor(raw_targets[label])) for label in labels
        }

        # Distribute remaining slots using largest remainder.
        remaining = n - sum(target_counts.values())
        order = sorted(
            labels,
            key=lambda label: (
                raw_targets[label] - target_counts[label],
                label,
            ),
            reverse=True,
        )
        for label in order[:remaining]:
            target_counts[label] += 1

        selected = []
        for label in labels:
            items = sorted(by_label[label], key=lambda x: x[0])
            rng.shuffle(items)
            selected.extend(items[:target_counts[label]])

        # Final deterministic ordering independent of dictionary iteration.
        selected.sort(key=lambda x: x[0])

    result = [(item[1], item[2], item[3]) for item in selected]

    real_count = sum(label == "real" for _, label, _ in result)
    fake_count = sum(label == "fake" for _, label, _ in result)

    logging.info(
        "Selected %d videos: %d real, %d fake.",
        len(result),
        real_count,
        fake_count,
    )

    return result


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def expand_bbox(
    bbox: List[float],
    margin: float,
    image_shape: Tuple[int, int],
) -> Optional[List[int]]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))

    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
        return None

    return [x1, y1, x2, y2]


def get_mtcnn() -> MTCNN:
    if torch.cuda.is_available():
        try:
            test = torch.zeros(1, device="cuda")
            _ = test + 1
            device = "cuda"
        except RuntimeError:
            logging.warning("CUDA test failed. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    logging.info("Using MTCNN on %s", device)
    return MTCNN(keep_all=True, device=device)


def extract_and_crop_video(
    video_path: Path,
    output_dir: Path,
    label: str,
    category: str,
    mtcnn: MTCNN,
) -> List[dict]:
    """
    Sample, detect, track, crop, and save one video.

    Frames are written directly as face crops. This avoids keeping the
    intermediate full-frame images on disk.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Failed to open %s", video_path)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metadata: List[dict] = []
    frame_idx = 0
    prev_bbox: Optional[List[float]] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLING_INTERVAL != 0:
                frame_idx += 1
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                boxes, probs = mtcnn.detect(img_rgb)
            except Exception as exc:
                logging.debug(
                    "Face detection failed for %s frame %d: %s",
                    video_path.name,
                    frame_idx,
                    exc,
                )
                boxes = None

            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            bboxes = [[float(v) for v in box] for box in boxes]

            if prev_bbox is None:
                selected_bbox = max(
                    bboxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                )
            else:
                ious = [compute_iou(prev_bbox, box) for box in bboxes]
                max_iou = max(ious)
                if max_iou >= IOU_THRESHOLD:
                    selected_bbox = bboxes[ious.index(max_iou)]
                else:
                    selected_bbox = max(
                        bboxes,
                        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    )

            expanded = expand_bbox(
                selected_bbox,
                BBOX_EXPANSION_MARGIN,
                frame.shape,
            )

            if expanded is None:
                frame_idx += 1
                continue

            x1, y1, x2, y2 = expanded
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                frame_idx += 1
                continue

            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_name

            ok = cv2.imwrite(str(frame_path), face_crop)
            if not ok:
                logging.warning("Failed to save %s", frame_path)
                frame_idx += 1
                continue

            frame_metadata.append(
                {
                    "frame_path": str(frame_path),
                    "video_id": video_path.stem,
                    "target_id": video_path.stem,
                    "source_id": None,
                    "category": category,
                    "label": label,
                    "original_frame_number": frame_idx,
                    "timestamp": frame_idx / FPS_ASSUMPTION,
                    "split": "test",
                    "face_bbox": expanded,
                    "face_detected": True,
                }
            )

            prev_bbox = selected_bbox
            frame_idx += 1

    finally:
        cap.release()

    if len(frame_metadata) < MIN_USABLE_FRAMES_PER_VIDEO:
        shutil.rmtree(output_dir, ignore_errors=True)
        return []

    return frame_metadata


def write_metadata(
    out_root: Path,
    selected: List[Tuple[Path, str, str]],
    processed: List[dict],
    skipped: List[str],
    n: Optional[int],
    source_root: Path,
) -> None:
    manifests = out_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(processed, columns=MANIFEST_COLUMNS)
    df.to_csv(manifests / "manifest.csv", index=False)

    with (manifests / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    video_records = []
    processed_ids = {row["video_id"] for row in processed}

    for path, label, category in selected:
        video_records.append(
            {
                "video_id": path.stem,
                "video_path": str(path),
                "category": category,
                "label": label,
                "split": "test",
                "usable": path.stem in processed_ids,
                "usable_frame_count": int(
                    sum(row["video_id"] == path.stem for row in processed)
                ),
            }
        )

    with (manifests / "video_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(video_records, f, indent=2)

    config = {
        "source_dataset": "Celeb-DF v2",
        "source_root": str(source_root),
        "selection": {
            "official_test_list": "List_of_testing_videos.txt",
            "N": n,
            "subset_seed": SUBSET_SEED if n is not None else None,
            "selection_policy": (
                "official test set"
                if n is None
                else "deterministic stratified subset preserving official test real/fake ratio"
            ),
        },
        "preprocessing": {
            "frame_sampling_interval": FRAME_SAMPLING_INTERVAL,
            "fps_assumption": FPS_ASSUMPTION,
            "bbox_expansion_margin": BBOX_EXPANSION_MARGIN,
            "iou_threshold": IOU_THRESHOLD,
            "min_face_size": MIN_FACE_SIZE,
            "min_usable_frames_per_video": MIN_USABLE_FRAMES_PER_VIDEO,
            "face_detector": "MTCNN",
        },
        "output_convention": {
            "split": "test",
            "real_category": "Original",
            "fake_category": "Deepfakes",
            "frame_root": "frames/test/{real|fake}/{video_id}/",
            "manifest_schema": MANIFEST_COLUMNS,
        },
        "processed_video_count": len(processed_ids),
        "skipped_video_count": len(skipped),
        "skipped_videos": skipped,
    }

    with (manifests / "preprocessing_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess Celeb-DF v2 into the project's FF++-compatible face-frame format."
    )
    parser.add_argument(
        "--in",
        dest="input",
        required=True,
        type=Path,
        help="Celeb-DF v2 dataset root.",
    )
    parser.add_argument(
        "--out",
        dest="output",
        required=True,
        type=Path,
        help="Output directory for the processed Celeb-DF evaluation data.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help=(
            "Optional total number of videos to use. "
            "If omitted, all videos in the official Celeb-DF v2 test list are used."
        ),
    )
    args = parser.parse_args()

    setup_logging()

    root = find_dataset_root(args.input)
    out_root = args.output.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logging.info("Celeb-DF root: %s", root)
    logging.info("Output root: %s", out_root)

    test_names = read_test_list(root)
    logging.info("Official test list contains %d unique entries.", len(test_names))

    video_index = build_video_index(root)
    selected = select_videos(test_names, video_index, args.N)

    mtcnn = get_mtcnn()

    processed: List[dict] = []
    skipped: List[str] = []

    for video_path, label, category in tqdm(
        selected,
        desc="Preprocessing Celeb-DF",
    ):
        output_dir = (
            out_root
            / "frames"
            / "test"
            / label
            / video_path.stem
        )

        rows = extract_and_crop_video(
            video_path,
            output_dir,
            label,
            category,
            mtcnn,
        )

        if len(rows) < MIN_USABLE_FRAMES_PER_VIDEO:
            skipped.append(video_path.stem)
            logging.warning(
                "Skipping %s: only %d usable frames.",
                video_path.name,
                len(rows),
            )
            continue

        processed.extend(rows)

    write_metadata(
        out_root,
        selected,
        processed,
        skipped,
        args.N,
        root,
    )

    processed_ids = sorted({row["video_id"] for row in processed})

    logging.info("Preprocessing complete.")
    logging.info("Selected videos: %d", len(selected))
    logging.info("Usable videos: %d", len(processed_ids))
    logging.info("Skipped videos: %d", len(skipped))
    logging.info("Usable frames: %d", len(processed))
    logging.info("Manifest: %s", out_root / "manifests" / "manifest.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
