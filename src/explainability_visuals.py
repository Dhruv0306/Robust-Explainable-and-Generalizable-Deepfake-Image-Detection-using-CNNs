"""
Approach 3 visual outputs.

Cache lookup is category-aware and therefore matches the corrected evaluation
identity:
(category, video_id, original_frame_number, transformation).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def _cache_path(
    cache_dir: Path,
    category: str,
    video_id: str,
    frame_number: int,
    transformation: str = "clean",
) -> Path:
    return (
        cache_dir
        / transformation
        / f"{category}_{video_id}_frame{frame_number:04d}.npy"
    )


def generate_global_heatmaps(
    frame_df: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
    target_resolution: Tuple[int, int] = (256, 256),
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_fakes = frame_df[
        (frame_df["transformation"].astype(str) == "clean")
        & (frame_df["ground_truth"] == 1)
    ]
    if clean_fakes.empty:
        return {}

    saved: Dict[str, Path] = {}
    categories = ["Overall"] + sorted(
        clean_fakes["category"].astype(str).unique()
    )

    for category in categories:
        subset = (
            clean_fakes
            if category == "Overall"
            else clean_fakes[
                clean_fakes["category"].astype(str) == category
            ]
        )

        accum = np.zeros(target_resolution, dtype=np.float64)
        valid = 0

        for _, row in subset.iterrows():
            path = _cache_path(
                cache_dir,
                str(row["category"]),
                str(row["video_id"]),
                int(row["original_frame_number"]),
            )
            if not path.exists():
                continue

            try:
                cam = np.load(path).astype(np.float32)
                cam = cv2.resize(
                    cam,
                    (target_resolution[1], target_resolution[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                accum += cam
                valid += 1
            except Exception as exc:
                logging.warning("Could not read %s: %s", path, exc)

        if valid == 0:
            continue

        mean_map = np.clip(accum / valid, 0.0, 1.0)
        vis = np.uint8(mean_map * 255.0)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        out = output_dir / f"global_heatmap_{category.lower()}.png"
        cv2.imwrite(str(out), vis)
        saved[category] = out

    return saved


def generate_representative_cases(
    frame_df: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    fake = frame_df[frame_df["ground_truth"] == 1].copy()
    clean = fake[fake["transformation"].astype(str) == "clean"].copy()

    if clean.empty or "SO" not in clean.columns:
        return generated

    clean = clean.dropna(subset=["SO"])
    if clean.empty:
        return generated

    for tag, row in [
        ("highest_SO", clean.loc[clean["SO"].idxmax()]),
        ("lowest_SO", clean.loc[clean["SO"].idxmin()]),
    ]:
        path = _save_clean_panel(
            row, cache_dir, output_dir, tag
        )
        if path:
            generated.append(path)

    # Add one prediction-change example when available.
    changed = fake[
        fake["prediction_state"].isin(
            ["Correct->Incorrect", "Incorrect->Correct"]
        )
    ]
    if not changed.empty:
        row = changed.iloc[0]
        path = _save_transformed_panel(
            row, cache_dir, output_dir, "prediction_changed"
        )
        if path:
            generated.append(path)

    return generated


def _save_clean_panel(
    row: pd.Series,
    cache_dir: Path,
    output_dir: Path,
    tag: str,
) -> Optional[Path]:
    try:
        category = str(row["category"])
        video_id = str(row["video_id"])
        frame_number = int(row["original_frame_number"])

        image = cv2.imread(str(row["frame_path"]))
        if image is None:
            return None

        cam_path = _cache_path(
            cache_dir,
            category,
            video_id,
            frame_number,
            "clean",
        )
        if not cam_path.exists():
            return None

        cam = np.load(cam_path).astype(np.float32)
        cam = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
        heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        h, w = image.shape[:2]
        if heat.shape[:2] != (h, w):
            heat = cv2.resize(heat, (w, h))

        blend = cv2.addWeighted(image, 0.60, heat, 0.40, 0)
        panel = np.hstack([image, heat, blend])

        out = output_dir / (
            f"case_{tag}_{category}_{video_id}_"
            f"frame{frame_number:04d}.png"
        )
        cv2.imwrite(str(out), panel)
        return out
    except Exception as exc:
        logging.warning("Failed to create %s case: %s", tag, exc)
        return None


def _save_transformed_panel(
    row: pd.Series,
    cache_dir: Path,
    output_dir: Path,
    tag: str,
) -> Optional[Path]:
    try:
        category = str(row["category"])
        video_id = str(row["video_id"])
        frame_number = int(row["original_frame_number"])
        transformation = str(row["transformation"])

        image = cv2.imread(str(row["frame_path"]))
        if image is None:
            return None

        cam_path = _cache_path(
            cache_dir,
            category,
            video_id,
            frame_number,
            transformation,
        )
        if not cam_path.exists():
            return None

        cam = np.load(cam_path).astype(np.float32)
        cam = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
        heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        h, w = image.shape[:2]
        if heat.shape[:2] != (h, w):
            heat = cv2.resize(heat, (w, h))

        blend = cv2.addWeighted(image, 0.60, heat, 0.40, 0)
        panel = np.hstack([image, heat, blend])

        out = output_dir / (
            f"case_{tag}_{category}_{video_id}_"
            f"frame{frame_number:04d}_{transformation}.png"
        )
        cv2.imwrite(str(out), panel)
        return out
    except Exception as exc:
        logging.warning(
            "Failed to create transformed case: %s", exc
        )
        return None
