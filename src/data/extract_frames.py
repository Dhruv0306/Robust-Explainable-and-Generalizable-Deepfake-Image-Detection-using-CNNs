"""
Extract frames from videos: every 4th frame at 30 FPS assumption.
"""
from pathlib import Path
import logging
import cv2
import json
from typing import Dict, List
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval: int = FRAME_SAMPLING_INTERVAL
) -> List[Dict]:
    """
    Extract every Nth frame from video.
    Returns list of frame metadata dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open {video_path}")
        return []

    frame_metadata = []
    frame_idx = 0
    saved_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_idx % interval == 0:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            # Metadata
            frame_metadata.append({
                "frame_path": str(frame_path),
                "original_frame_number": frame_idx,
                "timestamp": frame_idx / FPS_ASSUMPTION,
            })
            saved_count += 1

        frame_idx += 1

    cap.release()
    return frame_metadata


def extract_all_frames(video_paths: Dict[str, List[Path]], splits: Dict[str, List[str]]) -> Dict:
    """
    Extract frames from all videos, organize by split and category.
    Returns metadata dict.
    """
    logging.info("Extracting frames from all videos...")

    # Build video_id -> split mapping
    video_id_to_split = {}
    for split_name, video_ids in splits.items():
        for video_id in video_ids:
            video_id_to_split[video_id] = split_name

    all_metadata = []

    for category, paths in video_paths.items():
        logging.info(f"Processing {category}...")
        label = "real" if category == "Original" else "fake"

        for video_path in tqdm(paths, desc=category):
            video_id = video_path.stem.split("_")[0]  # Extract target ID
            split = video_id_to_split.get(video_id)

            if split is None:
                logging.warning(f"Video {video_id} not in any split, skipping")
                continue

            # Output directory: frames/<split>/<label>/<video_id>/
            output_dir = FRAMES_ROOT / split / label / video_path.stem

            # Extract frames
            frame_metadata = extract_frames_from_video(video_path, output_dir)

            # Add video-level metadata
            for fm in frame_metadata:
                fm.update({
                    "video_id": video_path.stem,
                    "target_id": video_id,
                    "source_id": video_path.stem.split("_")[1] if "_" in video_path.stem else None,
                    "category": category,
                    "label": label,
                    "split": split,
                })

            all_metadata.extend(frame_metadata)

    logging.info(f"Extracted {len(all_metadata)} total frames")
    return all_metadata


if __name__ == "__main__":
    from utils import setup_logging
    from inspect_dataset import inspect_dataset
    import json

    setup_logging()

    # Load splits
    with open(SPLITS_ROOT / "splits.json") as f:
        splits = json.load(f)

    video_paths = inspect_dataset()
    metadata = extract_all_frames(video_paths, splits)

    # Save frame extraction metadata
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS_ROOT / "frame_extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Frame extraction complete")
