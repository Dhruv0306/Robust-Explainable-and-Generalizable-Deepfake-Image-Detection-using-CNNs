"""
Visual Quality Verification and Example Generation.

Implements Plan Sections 41 & 42:
- Generates side-by-side visual panels: Clean | Severity 1 | Severity 2 | Severity 3.
- Verifies that corruptions scale monotonically and preserve visual interpretability.
- Saves candidate inspection images to figures/visual_examples/.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd

from robustness import (
    apply_brightness,
    apply_contrast,
    apply_crop,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg,
    apply_resize,
)


def create_labeled_tile(image_rgb: np.ndarray, label: str, font_scale: float = 0.45) -> np.ndarray:
    """Add label banner to top of an image tile."""
    tile = image_rgb.copy()
    h, w = tile.shape[:2]

    # Create banner at top
    banner_h = max(22, int(h * 0.12))
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    # Draw text in white
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = max(4, (w - tw) // 2)
    ty = (banner_h + th) // 2 - 2

    cv2.putText(banner, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    combined = np.vstack([banner, tile])
    return combined


def generate_transformation_strip(
    image_rgb: np.ndarray,
    transformation_name: str,
    severities: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Generate horizontal strip: [ Clean | Sev 1 | Sev 2 | Sev 3 ].

    Args:
        image_rgb: Original RGB face crop (H, W, 3).
        transformation_name: Name of corruption.
        severities: List of 3 dicts containing 'label' and 'func' callable.
    """
    tiles = [create_labeled_tile(image_rgb, "Clean (Sev 0)")]

    for s in severities:
        corrupted = s["func"](image_rgb)
        tiles.append(create_labeled_tile(corrupted, s["label"]))

    # Stack horizontally with 2px vertical separator
    sep = np.full((tiles[0].shape[0], 2, 3), 200, dtype=np.uint8)
    strip_parts = []
    for i, t in enumerate(tiles):
        strip_parts.append(t)
        if i < len(tiles) - 1:
            strip_parts.append(sep)

    return np.hstack(strip_parts)


def generate_visual_examples_for_image(
    image_rgb: np.ndarray,
    output_dir: Path,
    image_id: str,
    config: Dict[str, Any],
) -> List[Path]:
    """
    Generate visual comparison strips for all core enabled transformations on a single image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1. JPEG Compression
    if config.get("jpeg", {}).get("enabled", True):
        q = config["jpeg"]["qualities"]
        strip = generate_transformation_strip(
            image_rgb,
            "jpeg",
            [
                {"label": f"JPEG Q={q[0]}", "func": lambda img: apply_jpeg(img, q[0])},
                {"label": f"JPEG Q={q[1]}", "func": lambda img: apply_jpeg(img, q[1])},
                {"label": f"JPEG Q={q[2]}", "func": lambda img: apply_jpeg(img, q[2])},
            ],
        )
        p = output_dir / f"{image_id}_jpeg.png"
        cv2.imwrite(str(p), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        saved_paths.append(p)

    # 2. Lower-resolution Resizing
    if config.get("resize", {}).get("enabled", True):
        s = config["resize"]["scales"]
        strip = generate_transformation_strip(
            image_rgb,
            "resize",
            [
                {"label": f"Scale {int(s[0]*100)}%", "func": lambda img: apply_resize(img, s[0])},
                {"label": f"Scale {int(s[1]*100)}%", "func": lambda img: apply_resize(img, s[1])},
                {"label": f"Scale {int(s[2]*100)}%", "func": lambda img: apply_resize(img, s[2])},
            ],
        )
        p = output_dir / f"{image_id}_resize.png"
        cv2.imwrite(str(p), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        saved_paths.append(p)

    # 3. Brightness: Darkening
    if config.get("brightness", {}).get("enabled", True):
        d = config["brightness"]["darker"]
        strip = generate_transformation_strip(
            image_rgb,
            "brightness_dark",
            [
                {"label": f"Dark f={d[0]:.2f}", "func": lambda img: apply_brightness(img, d[0])},
                {"label": f"Dark f={d[1]:.2f}", "func": lambda img: apply_brightness(img, d[1])},
                {"label": f"Dark f={d[2]:.2f}", "func": lambda img: apply_brightness(img, d[2])},
            ],
        )
        p = output_dir / f"{image_id}_brightness_dark.png"
        cv2.imwrite(str(p), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        saved_paths.append(p)

    # 4. Brightness: Brightening
    if config.get("brightness", {}).get("enabled", True):
        b = config["brightness"]["brighter"]
        strip = generate_transformation_strip(
            image_rgb,
            "brightness_bright",
            [
                {"label": f"Bright f={b[0]:.2f}", "func": lambda img: apply_brightness(img, b[0])},
                {"label": f"Bright f={b[1]:.2f}", "func": lambda img: apply_brightness(img, b[1])},
                {"label": f"Bright f={b[2]:.2f}", "func": lambda img: apply_brightness(img, b[2])},
            ],
        )
        p = output_dir / f"{image_id}_brightness_bright.png"
        cv2.imwrite(str(p), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        saved_paths.append(p)

    return saved_paths


def generate_pilot_visual_examples(
    manifest_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    max_samples: int = 5,
) -> List[Path]:
    """
    Select representative face crops covering different manipulation categories
    and generate visual verification strips.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Pick one sample per unique category for diversity
    categories = test_df["category"].unique()
    selected_indices = []
    for cat in categories:
        cat_matches = test_df[test_df["category"] == cat].index
        if len(cat_matches) > 0:
            # Pick a middle frame from the category
            selected_indices.append(cat_matches[len(cat_matches) // 2])

    # If still fewer than max_samples, sample periodically (every 50th frame per Plan Section 41)
    if len(selected_indices) < max_samples:
        for idx in range(0, len(test_df), 50):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_samples:
                break

    all_saved = []
    for idx in selected_indices[:max_samples]:
        row = test_df.iloc[idx]
        frame_path = row["frame_path"]
        vid = row["video_id"]
        cat = row["category"]

        bgr = cv2.imread(frame_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        sample_id = f"{cat}_{vid}_frame{idx:04d}"
        saved = generate_visual_examples_for_image(rgb, output_dir, sample_id, config)
        all_saved.extend(saved)

    return all_saved
