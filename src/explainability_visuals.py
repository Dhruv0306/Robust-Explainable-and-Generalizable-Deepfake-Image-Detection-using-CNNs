"""
Approach 3: Global Heatmaps and Visual Case Extraction Suite.
Generates:
1. Overall and category-level aggregated Grad-CAM heatmaps
2. Predefined representative visual panels:
   - Correct fake vs. Missed fake
   - Strongest localization (highest SO) vs. Weakest localization (lowest SO)
   - Robustness clean vs. transformed explanation panels
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd


def generate_global_heatmaps(
    frame_df: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
    target_resolution: Tuple[int, int] = (256, 256),
) -> Dict[str, Path]:
    """
    Generate mean Grad-CAM heatmaps across all manipulated images and per manipulation category.

    Args:
        frame_df: DataFrame with frame_level_results.csv (clean pass)
        cache_dir: Path to cache/gradcam/clean/
        output_dir: Path to write global heatmap figures
        target_resolution: common (H, W) for spatial averaging

    Returns:
        Dict mapping category name to saved heatmap PNG path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_heatmaps: Dict[str, Path] = {}

    clean_fakes = frame_df[(frame_df["transformation"] == "clean") & (frame_df["ground_truth"] == 1)]
    if clean_fakes.empty:
        logging.warning("No clean fake frames found for global heatmap generation.")
        return saved_heatmaps

    categories = ["Overall"] + list(clean_fakes["category"].unique())

    for cat in categories:
        if cat == "Overall":
            cat_df = clean_fakes
        else:
            cat_df = clean_fakes[clean_fakes["category"] == cat]

        accum = np.zeros(target_resolution, dtype=np.float64)
        valid_count = 0

        for _, row in cat_df.iterrows():
            vid = str(row["video_id"])
            fnum = int(row["original_frame_number"])
            npy_path = cache_dir / "clean" / f"{vid}_frame{fnum:04d}.npy"

            if npy_path.exists():
                try:
                    cam = np.load(npy_path).astype(np.float32)
                    resized = cv2.resize(cam, (target_resolution[1], target_resolution[0]), interpolation=cv2.INTER_LINEAR)
                    accum += resized
                    valid_count += 1
                except Exception:
                    continue

        if valid_count > 0:
            mean_map = (accum / valid_count).astype(np.float32)
            # Normalize to [0, 255] for visual export
            vis_map = np.uint8(255 * np.clip(mean_map, 0.0, 1.0))
            vis_color = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

            out_path = output_dir / f"global_heatmap_{cat.lower()}.png"
            cv2.imwrite(str(out_path), vis_color)
            saved_heatmaps[cat] = out_path
            logging.info(f"Saved {cat} global heatmap ({valid_count} frames) -> {out_path}")

    return saved_heatmaps


def generate_representative_cases(
    frame_df: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> List[Path]:
    """
    Automatically extract predefined representative case panels:
    - Strongest SO (highest localization agreement)
    - Weakest SO (lowest localization agreement)
    - Prediction-preserved vs prediction-changed under corruptions

    Returns:
        List of generated panel paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    fakes = frame_df[frame_df["ground_truth"] == 1]
    if fakes.empty:
        return generated

    # 1. Clean Localization Extremes
    clean_fakes = fakes[fakes["transformation"] == "clean"]
    if not clean_fakes.empty and "SO" in clean_fakes.columns:
        sorted_so = clean_fakes.sort_values(by="SO", ascending=False)
        best_row = sorted_so.iloc[0]
        worst_row = sorted_so.iloc[-1]

        for tag, row in [("highest_SO", best_row), ("lowest_SO", worst_row)]:
            p = _save_single_clean_panel(row, cache_dir, output_dir, tag=tag)
            if p:
                generated.append(p)

    return generated


def _save_single_clean_panel(
    row: pd.Series,
    cache_dir: Path,
    output_dir: Path,
    tag: str,
) -> Optional[Path]:
    """Render and save a multi-panel visual figure: [Face Crop | Grad-CAM | Heatmap Blend]."""
    try:
        vid = str(row["video_id"])
        fnum = int(row["original_frame_number"])
        cat = str(row["category"])

        img_bgr = cv2.imread(row["frame_path"])
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]

        npy_path = cache_dir / "clean" / f"{vid}_frame{fnum:04d}.npy"
        if not npy_path.exists():
            return None

        cam = np.load(npy_path)
        cam_vis = np.uint8(255 * np.clip(cam, 0.0, 1.0))
        cam_color = cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)

        if cam_color.shape[:2] != (h, w):
            cam_color = cv2.resize(cam_color, (w, h))

        blended = cv2.addWeighted(img_bgr, 0.6, cam_color, 0.4, 0)
        panel = np.hstack([img_bgr, cam_color, blended])

        so_val = row.get("SO", 0.0)
        out_name = f"case_{tag}_{cat}_{vid}_frame{fnum:04d}_SO{so_val:.2f}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), panel)
        return out_path
    except Exception as e:
        logging.warning(f"Failed to generate case panel for {tag}: {e}")
        return None
