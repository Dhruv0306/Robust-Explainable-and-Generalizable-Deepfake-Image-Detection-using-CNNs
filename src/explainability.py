"""
Approach 3: Explainability and Explanation Robustness Main Orchestrator.
Coordinates:
1. Automated best-seed selection or manual seed override
2. Clean baseline inference and Grad-CAM generation
3. Robustness transformations (all 12 Approach 2 corruptions)
4. Mask loading and alignment across all 4 FF++ manipulation categories
5. Localization, stability, and intervention-based faithfulness evaluation
6. Video-level aggregation and statistical testing (Wilcoxon, bootstrap CIs, FDR)
7. Global heatmaps and representative visual case extraction
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Append src to path
SRC_ROOT = Path(__file__).parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from config import DATASET_ROOT, MANIFESTS_ROOT, OUTPUT_ROOT, CHECKPOINT_ROOT
from explainability_config import (
    EXPLAINABILITY_MODELS,
    CORE_ROBUSTNESS_CONDITIONS,
    PRIMARY_SALIENCY_THRESHOLD,
    SENSITIVITY_SALIENCY_THRESHOLDS,
    PRIMARY_FAITHFULNESS_METHOD,
    FAITHFULNESS_METHODS,
    EXPLANATION_STABILITY_THRESHOLD,
    FDR_FAMILIES,
    MIN_SAMPLE_WILCOXON,
    MIN_SAMPLE_SPEARMAN,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
)
from seed_selector import select_best_seed
from explainability_gradcam import GradCAMGenerator
from explainability_masks import MaskLoader
from explainability_metrics import (
    evaluate_frame_localization,
    evaluate_frame_stability,
    mask_salient_region,
    get_salient_mask,
)
from explainability_stats import (
    run_paired_wilcoxon_test,
    run_spearman_correlation,
    adjust_p_values_by_family,
)
from robustness import (
    apply_jpeg,
    apply_resize,
    apply_brightness,
)
from utils import get_device


def apply_transformation_by_condition(
    image_rgb: np.ndarray,
    cond_tuple: Tuple[str, str, int, Any],
) -> np.ndarray:
    """
    Apply Approach 2 transformation to uint8 RGB face crop.

    Args:
        image_rgb: (H, W, 3) uint8 array
        cond_tuple: (cond_name, family, severity, param_value)
    """
    cond_name, family, sev, param = cond_tuple

    if cond_name == "clean":
        return image_rgb.copy()
    elif family == "jpeg":
        return apply_jpeg(image_rgb, quality=int(param))
    elif family == "resize":
        return apply_resize(image_rgb, scale=float(param))
    elif family == "darkening" or family == "brightening":
        return apply_brightness(image_rgb, factor=float(param))
    else:
        raise ValueError(f"Unknown condition family: {family}")


class ExplainabilityOrchestrator:
    """
    Orchestrates end-to-end explainability evaluation for a single model and seed.
    """

    def __init__(
        self,
        model_name: str,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        pilot_mode: bool = False,
        conditions: Optional[List[Tuple[str, str, int, Any]]] = None,
    ):
        self.model_name = model_name
        self.pilot_mode = pilot_mode
        self.device = device if device is not None else get_device()[0]

        # Select best seed or apply manual override
        self.run_info = select_best_seed(
            model_name=self.model_name,
            output_dir=OUTPUT_ROOT,
            checkpoint_dir=CHECKPOINT_ROOT,
            target_seed=seed,
        )
        self.seed = self.run_info["seed"]

        # Define conditions to run
        if conditions is not None:
            self.conditions = conditions
        elif self.pilot_mode:
            # Pilot condition set (§47): clean, jpeg_q20, resize_025, dark_040, bright_160
            self.conditions = [
                ("clean", "clean", 0, None),
                ("jpeg", "jpeg", 3, 20),
                ("resize", "resize", 3, 0.25),
                ("brightness_dark", "darkening", 3, 0.40),
                ("brightness_bright", "brightening", 3, 1.60),
            ]
        else:
            self.conditions = CORE_ROBUSTNESS_CONDITIONS

        # Setup output directories
        self.output_dir = OUTPUT_ROOT / "explainability" / self.model_name / f"seed_{self.seed}"
        self.cache_dir = self.output_dir / "cache" / "gradcam"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Model, Grad-CAM generator, and Mask Loader
        self.generator = GradCAMGenerator(
            model_name=self.model_name,
            checkpoint_path=Path(self.run_info["checkpoint_path"]),
            device=self.device,
        )
        self.mask_loader = MaskLoader(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFESTS_ROOT / "manifest.csv",
        )

        # Load test manifest
        df = pd.read_csv(MANIFESTS_ROOT / "manifest.csv")
        self.test_df = df[df["split"] == "test"].copy()

        # In pilot mode, sample 5 frames per video across 1-2 videos per category
        if self.pilot_mode:
            pilot_vids = []
            for cat, group in self.test_df.groupby("category"):
                vids = group["video_id"].unique()
                pilot_vids.extend(vids[:2])
            sub = self.test_df[self.test_df["video_id"].isin(pilot_vids)].copy()
            # Sample first 5 frames per video for fast execution verification
            self.test_df = sub.groupby("video_id").head(5).copy()

        logging.info(
            f"Initialized {self.model_name} (seed {self.seed}): "
            f"{len(self.test_df)} frames across {self.test_df['video_id'].nunique()} videos. "
            f"Conditions: {len(self.conditions)} (pilot={self.pilot_mode})"
        )

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Execute full explainability evaluation across all conditions and frames.
        Streams rows to disk in batches of 100 to keep peak RAM usage bounded.
        """
        import gc
        FLUSH_BATCH = 100  # write to disk every N rows

        frame_csv_path = self.output_dir / "frame_level_results.csv"
        write_header = not frame_csv_path.exists()

        # Lightweight clean cache: only the minimal fields needed by transformed passes
        # (pred_fake, prob_fake for prediction-state; cam cache file already on disk)
        clean_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def _flush(rows: List[Dict[str, Any]], fh: Any, cols: List[str]) -> None:
            for rec in rows:
                fh.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")
            rows.clear()

        with open(frame_csv_path, "a", newline="", encoding="utf-8") as fh:
            # Determine fieldnames from a probe record (all keys appear in clean pass)
            probe_row = self.test_df.iloc[0]
            probe_rec = self._process_frame(probe_row, ("clean", "clean", 0, None), clean_ref=None)
            fieldnames = list(probe_rec.keys())

            if write_header:
                fh.write(",".join(fieldnames) + "\n")

            # ── 1. Clean Baseline Pass ─────────────────────────────────────────
            logging.info(f"--- Running Clean Baseline Pass for {self.model_name} (seed {self.seed}) ---")
            clean_cond = ("clean", "clean", 0, None)
            batch: List[Dict[str, Any]] = [probe_rec]
            key0 = (str(probe_row["video_id"]), int(probe_row["original_frame_number"]))
            clean_cache[key0] = {"pred_fake": probe_rec["pred_fake"], "prob_fake": probe_rec["prob_fake"]}

            for _, row in tqdm(
                self.test_df.iloc[1:].iterrows(),
                total=len(self.test_df) - 1,
                desc="Clean Pass",
            ):
                res = self._process_frame(row, clean_cond, clean_ref=None)
                batch.append(res)
                key = (str(row["video_id"]), int(row["original_frame_number"]))
                clean_cache[key] = {"pred_fake": res["pred_fake"], "prob_fake": res["prob_fake"]}

                if len(batch) >= FLUSH_BATCH:
                    _flush(batch, fh, fieldnames)
                    fh.flush()
                    gc.collect()

            if batch:
                _flush(batch, fh, fieldnames)
                fh.flush()
            torch.cuda.empty_cache()

            # ── 2. Transformed Conditions Passes ──────────────────────────────
            for cond_tuple in self.conditions:
                cond_name, family, sev, param = cond_tuple
                if cond_name == "clean":
                    continue

                cond_tag = f"{family}_sev{sev}" if sev > 0 else cond_name
                logging.info(f"--- Running Condition: {cond_tag} (param={param}) ---")

                for _, row in tqdm(self.test_df.iterrows(), total=len(self.test_df), desc=cond_tag):
                    key = (str(row["video_id"]), int(row["original_frame_number"]))
                    clean_ref = clean_cache.get(key)
                    res = self._process_frame(row, cond_tuple, clean_ref=clean_ref)
                    batch.append(res)

                    if len(batch) >= FLUSH_BATCH:
                        _flush(batch, fh, fieldnames)
                        fh.flush()
                        gc.collect()

                if batch:
                    _flush(batch, fh, fieldnames)
                    fh.flush()
                torch.cuda.empty_cache()

        # Free clean cache memory
        clean_cache.clear()
        gc.collect()
        logging.info(f"Frame-level CSV complete: {frame_csv_path}")

        # ── 3. Video-Level Aggregation (read streamed CSV) ────────────────────
        frame_df = pd.read_csv(frame_csv_path, low_memory=False)
        video_csv_path = self.output_dir / "video_level_results.csv"
        video_df = self._aggregate_to_video_level(frame_df)
        del frame_df
        gc.collect()
        video_df.to_csv(video_csv_path, index=False)
        logging.info(f"Saved video-level results ({len(video_df)} rows) to: {video_csv_path}")

        # ── 4. Statistical Analysis ───────────────────────────────────────────
        stats_csv_path = self.output_dir / "statistics_results.csv"
        stats_results = self._run_statistical_analysis(video_df)
        stats_results.to_csv(stats_csv_path, index=False)
        logging.info(f"Saved statistical test results ({len(stats_results)} rows) to: {stats_csv_path}")

        # ── 5. Global Summary Metadata ────────────────────────────────────────
        summary = {
            "model": self.model_name,
            "seed": int(self.seed),
            "pilot_mode": self.pilot_mode,
            "total_frames_evaluated": int(video_df["n_valid_frames"].sum()) if "n_valid_frames" in video_df else len(self.test_df),
            "total_videos": int(video_df["video_id"].nunique()),
            "conditions_evaluated": [f"{c[1]}_{c[2]}" if c[2] > 0 else c[0] for c in self.conditions],
            "checkpoint_path": self.run_info["checkpoint_path"],
            "target_layers": [str(l) for l in self.generator.target_layers],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # ── 6. Visuals ────────────────────────────────────────────────────────
        try:
            from explainability_visuals import generate_global_heatmaps, generate_representative_cases
            frame_df_vis = pd.read_csv(frame_csv_path, low_memory=False)
            figs_dir = self.output_dir / "figures"
            generate_global_heatmaps(frame_df_vis, self.cache_dir, figs_dir)
            generate_representative_cases(frame_df_vis, self.cache_dir, figs_dir)
            del frame_df_vis
            gc.collect()
        except Exception as e:
            logging.warning(f"Failed to generate visual figures: {e}")

        self.mask_loader.close()
        return summary

    def _process_frame(
        self,
        row: pd.Series,
        cond_tuple: Tuple[str, str, int, Any],
        clean_ref: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image frame for a given condition.
        """
        cond_name, family, sev, param = cond_tuple
        cond_tag = f"{family}_sev{sev}" if sev > 0 else cond_name

        vid = str(row["video_id"])
        fnum = int(row["original_frame_number"])
        cat = str(row["category"])
        gt_label = 1 if row["label"] == "fake" else 0

        # Load raw face crop
        img_bgr = cv2.imread(row["frame_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Apply Approach 2 transformation
        transformed_rgb = apply_transformation_by_condition(img_rgb, cond_tuple)

        # 1. Forward inference on transformed image
        prob_fake, pred_fake = self.generator.predict(transformed_rgb)

        record: Dict[str, Any] = {
            "model": self.model_name,
            "seed": self.seed,
            "video_id": vid,
            "original_frame_number": fnum,
            "frame_path": str(row["frame_path"]),
            "category": cat,
            "ground_truth": gt_label,
            "transformation": cond_tag,
            "transformation_family": family,
            "severity": sev,
            "prob_fake": prob_fake,
            "pred_fake": pred_fake,
            "correct": int(pred_fake == gt_label),
        }

        # 2. Grad-CAM generation and explainability (Evaluated on fake frames per Plan §3.2)
        if gt_label == 1:
            cache_file = self.cache_dir / cond_tag / f"{vid}_frame{fnum:04d}.npy"
            cam_map = self.generator.get_or_generate_cam(
                transformed_rgb,
                cache_path=cache_file,
                target_shape=(orig_h, orig_w),
            )

            # 3. Ground-truth Mask & Localization
            gt_mask = self.mask_loader.load_face_crop_mask(
                category=cat,
                video_id=vid,
                original_frame_number=fnum,
                target_shape=(orig_h, orig_w),
            )
            if gt_mask is not None:
                loc_res = evaluate_frame_localization(cam_map, gt_mask)
                record.update(loc_res)

            # 4. Intervention-based Faithfulness
            s_mask_20 = get_salient_mask(cam_map, top_fraction=PRIMARY_SALIENCY_THRESHOLD)
            for m_method in FAITHFULNESS_METHODS:
                masked_img = mask_salient_region(transformed_rgb, s_mask_20, method=m_method)
                p_masked, _ = self.generator.predict(masked_img)
                record[f"prob_fake_{m_method}"] = p_masked
                record[f"faithfulness_{m_method}"] = prob_fake - p_masked

            # Primary alias
            record["faithfulness"] = record.get(f"faithfulness_{PRIMARY_FAITHFULNESS_METHOD}", np.nan)

            # 5. Stability against Clean Reference
            if clean_ref is not None:
                clean_cache_file = self.cache_dir / "clean" / f"{vid}_frame{fnum:04d}.npy"
                if clean_cache_file.exists():
                    cam_clean = np.load(clean_cache_file)
                    stab_res = evaluate_frame_stability(cam_clean, cam_map)
                    record.update(stab_res)

        # Prediction state transitions (tracked across all frames if clean_ref provided)
        if clean_ref is not None:
            c_pred = clean_ref["pred_fake"]
            t_pred = pred_fake
            record["clean_prob_fake"] = clean_ref["prob_fake"]
            record["clean_pred_fake"] = c_pred
            record["prediction_preserved"] = int(c_pred == t_pred)

            if gt_label == 1:
                if c_pred == 1 and t_pred == 1:
                    record["prediction_state"] = "Correct->Correct"
                elif c_pred == 1 and t_pred == 0:
                    record["prediction_state"] = "Correct->Incorrect"
                elif c_pred == 0 and t_pred == 1:
                    record["prediction_state"] = "Incorrect->Correct"
                else:
                    record["prediction_state"] = "Incorrect->Incorrect"

        return record

    def _aggregate_to_video_level(self, frame_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate scalar metrics from frame to video level using mean aggregation.
        """
        group_cols = [
            "model",
            "seed",
            "video_id",
            "category",
            "ground_truth",
            "transformation",
            "transformation_family",
            "severity",
        ]

        metric_cols = [
            "prob_fake",
            "pred_fake",
            "correct",
            "SO",
            "SO@10",
            "SO@20",
            "SO@30",
            "IoU",
            "IoU@10",
            "IoU@20",
            "IoU@30",
            "saliency_mass",
            "hit_rate",
            "saliency_entropy",
            "faithfulness",
            "faithfulness_blur",
            "faithfulness_zero",
            "faithfulness_mean",
            "ES_cos",
            "explanation_IoU",
            "explanation_IoU@10",
            "explanation_IoU@20",
            "explanation_IoU@30",
            "prediction_preserved",
        ]

        existing_metrics = [c for c in metric_cols if c in frame_df.columns]
        agg_dict = {col: "mean" for col in existing_metrics}
        agg_dict["original_frame_number"] = "count"

        video_df = frame_df.groupby(group_cols, as_index=False).agg(agg_dict)
        video_df.rename(columns={"original_frame_number": "n_valid_frames"}, inplace=True)
        video_df["n_failed_frames"] = 0

        return video_df

    def _run_statistical_analysis(self, video_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute paired Wilcoxon signed-rank tests and Benjamini-Hochberg FDR correction.
        """
        stat_rows: List[Dict[str, Any]] = []

        # Only evaluate manipulated fake videos for localization & faithfulness
        fake_videos = video_df[video_df["ground_truth"] == 1]
        clean_fake = fake_videos[fake_videos["transformation"] == "clean"].set_index("video_id")

        for cond in fake_videos["transformation"].unique():
            if cond == "clean":
                continue

            trans_subset = fake_videos[fake_videos["transformation"] == cond].set_index("video_id")
            common_vids = clean_fake.index.intersection(trans_subset.index)

            if len(common_vids) == 0:
                continue

            c_sub = clean_fake.loc[common_vids]
            t_sub = trans_subset.loc[common_vids]

            # Family 1: Localization
            for metric in ["SO", "IoU", "saliency_mass", "hit_rate"]:
                if metric in c_sub.columns and metric in t_sub.columns:
                    w = run_paired_wilcoxon_test(c_sub[metric].values, t_sub[metric].values)
                    w.update({
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": metric,
                        "test_family": "family_1_localization",
                    })
                    stat_rows.append(w)

            # Family 2: Faithfulness
            for f_method in ["faithfulness_blur", "faithfulness_zero", "faithfulness_mean"]:
                if f_method in c_sub.columns and f_method in t_sub.columns:
                    w = run_paired_wilcoxon_test(c_sub[f_method].values, t_sub[f_method].values)
                    w.update({
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": f_method,
                        "test_family": "family_2_faithfulness",
                    })
                    stat_rows.append(w)

        stats_df = pd.DataFrame(stat_rows)
        if not stats_df.empty:
            stats_df = adjust_p_values_by_family(stats_df)
        return stats_df


def main():
    parser = argparse.ArgumentParser(description="Approach 3: Explainability and Explanation Robustness")
    parser.add_argument("--model", type=str, default="xception", choices=EXPLAINABILITY_MODELS, help="Model architecture")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed override (default: auto best seed)")
    parser.add_argument("--check-all", action="store_true", help="Run evaluation across all 3 architectures independently")
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode on small representative subset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dev, dev_info = get_device()
    logging.info(f"Starting Approach 3 Orchestrator on {dev} ({dev_info['device_name']})")

    models_to_run = EXPLAINABILITY_MODELS if args.check_all else [args.model]

    if args.check_all and args.seed is not None:
        raise ValueError("Cannot pass --seed with --check-all. Seeds must be resolved independently per architecture.")

    for m in models_to_run:
        logging.info(f"========== Processing Architecture: {m.upper()} ==========")
        orchestrator = ExplainabilityOrchestrator(
            model_name=m,
            seed=args.seed,
            device=dev,
            pilot_mode=args.pilot,
        )
        orchestrator.run_evaluation()

    logging.info("All explainability runs completed successfully.")


if __name__ == "__main__":
    main()
