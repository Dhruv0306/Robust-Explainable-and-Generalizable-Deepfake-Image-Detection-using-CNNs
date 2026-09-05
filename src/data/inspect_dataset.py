"""
Dataset inspection: verify video counts and collect metadata
"""
from pathlib import Path
import logging
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def inspect_dataset() -> Dict[str, List[Path]]:
    """
    Inspect FaceForensics++ dataset structure.
    Returns dict mapping category -> list of video paths.
    """
    logging.info("Inspecting FaceForensics++ dataset...")

    video_paths = {
        "Original": list(ORIGINAL_VIDEOS.glob("*.mp4")),
        "Deepfakes": list(DEEPFAKES_VIDEOS.glob("*.mp4")),
        "Face2Face": list(FACE2FACE_VIDEOS.glob("*.mp4")),
        "FaceSwap": list(FACESWAP_VIDEOS.glob("*.mp4")),
        "NeuralTextures": list(NEURALTEXTURES_VIDEOS.glob("*.mp4")),
    }

    logging.info("Video counts per category:")
    for cat, paths in video_paths.items():
        logging.info(f"  {cat}: {len(paths)}")

    total = sum(len(p) for p in video_paths.values())
    logging.info(f"Total videos: {total}")

    # Verify counts
    assert total == TOTAL_VIDEOS, f"Expected {TOTAL_VIDEOS}, found {total}"
    for cat in video_paths:
        assert len(video_paths[cat]) == VIDEOS_PER_CATEGORY, \
            f"{cat}: expected {VIDEOS_PER_CATEGORY}, found {len(video_paths[cat])}"

    logging.info("Dataset inspection passed ✓")
    return video_paths


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    inspect_dataset()
