"""
Build CSV and JSON manifests from processed face crop metadata.
"""
from pathlib import Path
import logging
import json
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def build_manifest(face_processed_metadata: list[dict]):
    """
    Build final CSV and JSON manifests from face-processed metadata.
    """
    logging.info("Building manifests...")

    # Convert to DataFrame
    df = pd.DataFrame(face_processed_metadata)

    # Reorder columns
    column_order = [
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
    df = df[column_order]

    # Save CSV
    csv_path = MANIFESTS_ROOT / "manifest.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"CSV manifest saved: {csv_path}")

    # Save JSON
    json_path = MANIFESTS_ROOT / "manifest.json"
    with open(json_path, "w") as f:
        json.dump(face_processed_metadata, f, indent=2)
    logging.info(f"JSON manifest saved: {json_path}")

    # Log statistics
    logging.info(f"Total frames: {len(df)}")
    logging.info(f"Split distribution:")
    for split in ["train", "val", "test"]:
        count = len(df[df["split"] == split])
        logging.info(f"  {split}: {count}")

    logging.info(f"Label distribution:")
    for label in ["real", "fake"]:
        count = len(df[df["label"] == label])
        logging.info(f"  {label}: {count}")

    logging.info(f"Category distribution:")
    for cat in df["category"].unique():
        count = len(df[df["category"] == cat])
        logging.info(f"  {cat}: {count}")

    return df


if __name__ == "__main__":
    from utils import setup_logging

    setup_logging()

    # Load face-processed metadata
    with open(MANIFESTS_ROOT / "face_processed_metadata.json") as f:
        metadata = json.load(f)

    build_manifest(metadata)
    logging.info("Manifest build complete")
