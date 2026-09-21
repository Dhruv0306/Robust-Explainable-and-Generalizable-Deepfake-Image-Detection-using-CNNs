"""
Approach 3: FaceForensics++ Ground-Truth Manipulation Mask Loader and Alignment Pipeline.
Extracts full-frame masks from compressed mp4 videos, aligns them with face-crop coordinates,
handles category-level bounding-box lookup, and validates spatial registration.
"""
import ast
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import pandas as pd


class MaskLoader:
    """
    Loads, caches, and aligns FaceForensics++ manipulation masks with face crops.
    """

    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Optional[Path] = None,
    ):
        """
        Initialize MaskLoader with dataset paths and precomputed bbox lookup.

        Args:
            dataset_root: Path to data/datasets/FaceForensics++
            manifest_path: Path to data/manifests/manifest.csv
        """
        self.dataset_root = Path(dataset_root)
        self.manifest_path = manifest_path
        self._video_caps: Dict[str, cv2.VideoCapture] = {}
        self.bbox_lookup: Dict[Tuple[str, int], List[int]] = {}
        self._mask_crop_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

        if self.manifest_path is not None and self.manifest_path.exists():
            self._build_bbox_lookup()

    def _build_bbox_lookup(self) -> None:
        """
        Build a robust (video_id, original_frame_number) -> [x1, y1, x2, y2] lookup.
        Filters out dummy [0, 0, W, H] boxes by prioritizing genuine full-frame detections
        (e.g., from Deepfakes or detected original sequences).
        """
        logging.info(f"Building bounding box lookup table from {self.manifest_path}...")
        df = pd.read_csv(self.manifest_path)
        test_fake = df[(df["split"] == "test") & (df["label"] == "fake")]

        for _, row in test_fake.iterrows():
            vid = str(row["video_id"])
            fnum = int(row["original_frame_number"])
            bbox_str = str(row["face_bbox"])

            try:
                bbox = ast.literal_eval(bbox_str)
                # Genuine box check: not a dummy [0, 0, ...]
                if isinstance(bbox, list) and len(bbox) == 4:
                    if bbox[0] != 0 or bbox[1] != 0:
                        self.bbox_lookup[(vid, fnum)] = bbox
                    elif (vid, fnum) not in self.bbox_lookup:
                        # Fallback only if no genuine box seen yet
                        self.bbox_lookup[(vid, fnum)] = bbox
            except Exception:
                continue

        logging.info(f"Bounding box lookup ready with {len(self.bbox_lookup)} verified frames.")

    def get_mask_video_path(self, category: str, video_id: str) -> Path:
        """Resolve path to ground-truth manipulation mask video."""
        return (
            self.dataset_root
            / "manipulated_sequences"
            / category
            / "masks"
            / "videos"
            / f"{video_id}.mp4"
        )

    def _get_capture(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        """Get or create cached cv2.VideoCapture instance with max open handle management."""
        vpath_str = str(video_path)
        if vpath_str not in self._video_caps:
            if not video_path.exists():
                logging.warning(f"Mask video missing: {video_path}")
                return None
            # Keep max 8 video handles open simultaneously to limit resource usage
            if len(self._video_caps) >= 8:
                oldest_key = next(iter(self._video_caps))
                old_cap = self._video_caps.pop(oldest_key)
                try:
                    old_cap.release()
                except Exception:
                    pass

            cap = cv2.VideoCapture(vpath_str)
            if not cap.isOpened():
                logging.warning(f"Failed to open mask video: {video_path}")
                return None
            self._video_caps[vpath_str] = cap

        return self._video_caps[vpath_str]

    def load_full_frame_mask(
        self,
        category: str,
        video_id: str,
        original_frame_number: int,
    ) -> Optional[np.ndarray]:
        """
        Extract full-frame manipulation mask frame from video.

        Returns:
            uint8 grayscale numpy array of shape (H, W), or None on failure
        """
        mask_path = self.get_mask_video_path(category, video_id)
        cap = self._get_capture(mask_path)
        if cap is None:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(original_frame_number))
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning(
                f"Failed to read frame {original_frame_number} from {mask_path}"
            )
            return None

        # Convert to grayscale
        if frame.ndim == 3:
            mask_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = frame

        return mask_gray

    def load_face_crop_mask(
        self,
        category: str,
        video_id: str,
        original_frame_number: int,
        target_shape: Optional[Tuple[int, int]] = None,
        face_bbox: Optional[List[int]] = None,
        threshold: int = 127,
    ) -> Optional[np.ndarray]:
        """
        Load and align ground-truth manipulation mask to face-crop coordinates.

        Args:
            category: 'Deepfakes', 'Face2Face', 'FaceSwap', or 'NeuralTextures'
            video_id: e.g. '033_097'
            original_frame_number: integer original frame index
            target_shape: (H, W) of corresponding face crop image
            face_bbox: optional [x1, y1, x2, y2]. If None, uses lookup table.
            threshold: binary binarization threshold (default 127)

        Returns:
            Binary uint8 numpy array of shape (H, W) with values in {0, 1}, or None
        """
        cache_key = (category, str(video_id), int(original_frame_number))
        if cache_key in self._mask_crop_cache:
            cached_bin = self._mask_crop_cache[cache_key]
            if target_shape is None or cached_bin.shape == target_shape:
                return cached_bin
            return cv2.resize(
                cached_bin.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)

        full_mask = self.load_full_frame_mask(category, video_id, original_frame_number)
        if full_mask is None:
            return None

        # Resolve genuine bounding box
        bbox = face_bbox
        if bbox is None or (bbox[0] == 0 and bbox[1] == 0):
            bbox = self.bbox_lookup.get((video_id, int(original_frame_number)), bbox)

        if bbox is None:
            logging.warning(f"No bounding box found for ({video_id}, {original_frame_number})")
            return None

        x1, y1, x2, y2 = bbox
        h_full, w_full = full_mask.shape[:2]

        # Defensive bounds clipping
        x1 = max(0, min(x1, w_full - 1))
        y1 = max(0, min(y1, h_full - 1))
        x2 = max(x1 + 1, min(x2, w_full))
        y2 = max(y1 + 1, min(y2, h_full))

        cropped_mask = full_mask[y1:y2, x1:x2]

        # Binarize mask
        bin_mask = (cropped_mask > threshold).astype(np.uint8)
        self._mask_crop_cache[cache_key] = bin_mask

        # Ensure exact match to target_shape if provided
        if target_shape is not None and bin_mask.shape != target_shape:
            bin_mask = cv2.resize(
                bin_mask.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)

        return bin_mask

    def close(self) -> None:
        """Release all open video captures."""
        for cap in self._video_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def validate_mask_alignment(
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Optional[Path] = None,
    samples_per_category: int = 3,
) -> Dict[str, Any]:
    """
    Mandatory Gate 4 verification suite:
    Validates face-crop and manipulation-mask alignment across all 4 categories.
    Saves visual alignment overlays if output_dir is provided.

    Returns:
        Summary dict containing validation metrics and status
    """
    logging.info("Running Mandatory Gate 4: Mask Alignment Validation...")
    df = pd.read_csv(manifest_path)
    test_fake = df[(df["split"] == "test") & (df["label"] == "fake")]

    loader = MaskLoader(dataset_root, manifest_path)
    report = {
        "checked_samples": 0,
        "successful_mappings": 0,
        "missing_masks": 0,
        "empty_masks": 0,
        "dimension_mismatches": 0,
        "failures": [],
        "categories_verified": [],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    categories = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for cat in categories:
        cat_df = test_fake[test_fake["category"] == cat]
        if cat_df.empty:
            logging.warning(f"No test samples for category: {cat}")
            continue

        sampled_rows = cat_df.sample(min(samples_per_category, len(cat_df)), random_state=42)

        for idx, (_, row) in enumerate(sampled_rows.iterrows()):
            report["checked_samples"] += 1
            vid = str(row["video_id"])
            fnum = int(row["original_frame_number"])
            frame_path = Path(row["frame_path"])

            # Read face crop image
            img_bgr = cv2.imread(str(frame_path))
            if img_bgr is None:
                report["failures"].append({"category": cat, "video_id": vid, "frame": fnum, "error": "missing_frame_image"})
                continue

            h, w = img_bgr.shape[:2]

            # Load mask
            mask = loader.load_face_crop_mask(
                category=cat,
                video_id=vid,
                original_frame_number=fnum,
                target_shape=(h, w),
            )

            if mask is None:
                report["missing_masks"] += 1
                report["failures"].append({"category": cat, "video_id": vid, "frame": fnum, "error": "mask_lookup_failed"})
                continue

            if mask.shape != (h, w):
                report["dimension_mismatches"] += 1
                report["failures"].append({"category": cat, "video_id": vid, "frame": fnum, "error": "dimension_mismatch"})
                continue

            report["successful_mappings"] += 1
            if mask.sum() == 0:
                report["empty_masks"] += 1

            # Save visual verification panel if output requested
            if output_dir is not None:
                overlay = img_bgr.copy()
                overlay[mask == 1] = (0, 0, 255)  # highlight mask in red
                blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
                mask_vis = (mask * 255).astype(np.uint8)
                mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

                panel = np.hstack([img_bgr, mask_vis_bgr, blended])
                out_name = f"align_gate4_{cat}_{vid}_frame{fnum:04d}.png"
                cv2.imwrite(str(output_dir / out_name), panel)

        report["categories_verified"].append(cat)

    loader.close()
    success_rate = report["successful_mappings"] / max(1, report["checked_samples"])
    report["success_rate"] = success_rate
    report["passed"] = (
        report["checked_samples"] > 0
        and report["dimension_mismatches"] == 0
        and report["missing_masks"] == 0
        and success_rate == 1.0
    )

    logging.info(
        f"Gate 4 Validation Complete: {report['successful_mappings']}/{report['checked_samples']} "
        f"mapped successfully (Passed={report['passed']})."
    )
    return report


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import DATASET_ROOT, MANIFESTS_ROOT, OUTPUT_ROOT

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest = MANIFESTS_ROOT / "manifest.csv"
    val_dir = OUTPUT_ROOT / "explainability" / "validation_gate4"

    res = validate_mask_alignment(
        dataset_root=DATASET_ROOT,
        manifest_path=manifest,
        output_dir=val_dir,
        samples_per_category=3,
    )
    print("\n=== GATE 4 VALIDATION REPORT ===")
    for k, v in res.items():
        if k != "failures":
            print(f"{k}: {v}")
    if res["failures"]:
        print("Failures:", res["failures"])
    assert res["passed"], "Gate 4 Validation FAILED!"
    print(f"\nVisual inspection panels saved to: {val_dir}")
