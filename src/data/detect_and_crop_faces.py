"""
Face detection and cropping using MTCNN (facenet-pytorch) with IoU-based tracking.
"""
from pathlib import Path
import logging
import cv2
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

try:
    from facenet_pytorch import MTCNN
    import torch
except ImportError:
    logging.warning("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
    MTCNN = None
    torch = None


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def expand_bbox(bbox: List[float], margin: float, img_shape: Tuple[int, int]) -> Optional[List[int]]:
    """
    Expand bbox by margin percentage, clip to image bounds.
    Returns None if invalid.
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox

    # Expand
    w_bbox = x2 - x1
    h_bbox = y2 - y1
    x1 = x1 - w_bbox * margin
    y1 = y1 - h_bbox * margin
    x2 = x2 + w_bbox * margin
    y2 = y2 + h_bbox * margin

    # Clip
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # Validate
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
        return None

    return [x1, y1, x2, y2]


def detect_and_crop_faces(frame_metadata: List[Dict]) -> List[Dict]:
    """
    Process frames for one video: detect faces, track via IoU, crop and save.
    Returns updated metadata with face crop info.
    """
    if not frame_metadata:
        return []

    # Initialize MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    processed = []
    prev_bbox = None

    for fm in frame_metadata:
        frame_path = Path(fm["frame_path"])
        if not frame_path.exists():
            logging.warning(f"Frame not found: {frame_path}")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            logging.warning(f"Failed to read {frame_path}")
            continue

        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces - returns boxes as [x1, y1, x2, y2], probs
        try:
            boxes, probs = mtcnn.detect(img_rgb)
        except Exception as e:
            logging.debug(f"Detection failed for {frame_path}: {e}")
            boxes, probs = None, None

        if boxes is None or len(boxes) == 0:
            # No face detected
            logging.debug(f"No face in {frame_path}")
            continue

        # Convert boxes to list format for processing
        bboxes = [[float(x) for x in box] for box in boxes]

        # Select face
        if prev_bbox is None:
            # First frame: select largest face
            selected_bbox = max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        else:
            # Track via IoU
            ious = [compute_iou(prev_bbox, b) for b in bboxes]
            max_iou = max(ious)
            if max_iou >= IOU_THRESHOLD:
                selected_bbox = bboxes[ious.index(max_iou)]
            else:
                # Lost tracking, re-init with largest
                selected_bbox = max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        # Expand bbox
        expanded = expand_bbox(selected_bbox, BBOX_EXPANSION_MARGIN, img.shape)
        if expanded is None:
            logging.debug(f"Invalid bbox for {frame_path}")
            continue

        # Crop face
        x1, y1, x2, y2 = expanded
        face_crop = img[y1:y2, x1:x2]

        # Save crop (overwrite original frame with face crop)
        cv2.imwrite(str(frame_path), face_crop)

        # Update metadata
        fm["face_bbox"] = expanded
        fm["face_detected"] = True
        processed.append(fm)
        prev_bbox = selected_bbox

    return processed


def process_all_videos(frame_extraction_metadata: List[Dict]) -> List[Dict]:
    """
    Group frames by video, process each video's faces.
    """
    logging.info("Detecting and cropping faces...")

    # Group by video_id
    video_groups = {}
    for fm in frame_extraction_metadata:
        vid = fm["video_id"]
        if vid not in video_groups:
            video_groups[vid] = []
        video_groups[vid].append(fm)

    all_processed = []
    skipped_videos = []

    for video_id, frames in tqdm(video_groups.items(), desc="Processing videos"):
        processed = detect_and_crop_faces(frames)

        if len(processed) < MIN_USABLE_FRAMES_PER_VIDEO:
            logging.warning(f"Video {video_id}: only {len(processed)} usable frames, skipping")
            skipped_videos.append(video_id)
            # Delete extracted frames for this video
            for fm in frames:
                fp = Path(fm["frame_path"])
                if fp.exists():
                    fp.unlink()
            continue

        all_processed.extend(processed)

    logging.info(f"Processed {len(all_processed)} frames from {len(video_groups) - len(skipped_videos)} videos")
    logging.info(f"Skipped {len(skipped_videos)} videos")

    return all_processed


if __name__ == "__main__":
    from utils import setup_logging
    import json

    setup_logging()

    if MTCNN is None:
        logging.error("facenet-pytorch not available. Install facenet-pytorch.")
        sys.exit(1)

    # Load frame extraction metadata
    with open(MANIFESTS_ROOT / "frame_extraction_metadata.json") as f:
        frame_metadata = json.load(f)

    processed = process_all_videos(frame_metadata)

    # Save processed metadata
    with open(MANIFESTS_ROOT / "face_processed_metadata.json", "w") as f:
        json.dump(processed, f, indent=2)

    logging.info("Face processing complete")
