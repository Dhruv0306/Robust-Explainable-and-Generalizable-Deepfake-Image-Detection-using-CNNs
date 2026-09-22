"""
Approach 3: Explainability and explanation robustness orchestrator.

Corrective version:
- uses a fresh output namespace;
- treats (category, video_id, original_frame_number) as the sample identity;
- validates exact condition coverage and duplicate-free evaluation keys;
- keeps category in all cache identities;
- performs primary paired inference at the source-video level;
- uses source-video aggregation across manipulation categories so the same
  source pair is not counted as four independent observations.
"""

import argparse
import csv
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

SRC_ROOT = Path(__file__).parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from config import DATASET_ROOT, MANIFESTS_ROOT, OUTPUT_ROOT, CHECKPOINT_ROOT
from explainability_config import (
    EXPLAINABILITY_MODELS,
    CORE_ROBUSTNESS_CONDITIONS,
    EXPLAINABILITY_OUTPUT_DIR,
    PRIMARY_SALIENCY_THRESHOLD,
    FAITHFULNESS_METHODS,
    PRIMARY_FAITHFULNESS_METHOD,
    EXPLANATION_STABILITY_THRESHOLD,
    MIN_SAMPLE_WILCOXON,
    MIN_SAMPLE_SPEARMAN,
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
    adjust_p_values_by_family,
)
from robustness import apply_jpeg, apply_resize, apply_brightness
from utils import get_device


def apply_transformation_by_condition(
    image_rgb: np.ndarray,
    cond_tuple: Tuple[str, str, int, Any],
) -> np.ndarray:
    cond_name, family, _, param = cond_tuple

    if cond_name == "clean":
        return image_rgb.copy()
    if family == "jpeg":
        return apply_jpeg(image_rgb, quality=int(param))
    if family == "resize":
        return apply_resize(image_rgb, scale=float(param))
    if family in ("darkening", "brightening"):
        return apply_brightness(image_rgb, factor=float(param))
    raise ValueError(f"Unknown condition family: {family}")


def condition_tag(cond_tuple: Tuple[str, str, int, Any]) -> str:
    cond_name, family, severity, _ = cond_tuple
    return f"{family}_sev{severity}" if severity > 0 else cond_name


class ExplainabilityOrchestrator:
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

        self.run_info = select_best_seed(
            model_name=model_name,
            output_dir=OUTPUT_ROOT,
            checkpoint_dir=CHECKPOINT_ROOT,
            target_seed=seed,
        )
        self.seed = int(self.run_info["seed"])

        if conditions is not None:
            self.conditions = conditions
        elif pilot_mode:
            self.conditions = [
                ("clean", "clean", 0, None),
                ("jpeg", "jpeg", 3, 20),
                ("resize", "resize", 3, 0.25),
                ("brightness_dark", "darkening", 3, 0.40),
                ("brightness_bright", "brightening", 3, 1.60),
            ]
        else:
            self.conditions = CORE_ROBUSTNESS_CONDITIONS

        self.output_dir = (
            OUTPUT_ROOT
            / EXPLAINABILITY_OUTPUT_DIR
            / self.model_name
            / f"seed_{self.seed}"
        )
        self.cache_dir = self.output_dir / "cache" / "gradcam"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.generator = GradCAMGenerator(
            model_name=self.model_name,
            checkpoint_path=Path(self.run_info["checkpoint_path"]),
            device=self.device,
        )
        self.mask_loader = MaskLoader(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFESTS_ROOT / "manifest.csv",
        )

        manifest = pd.read_csv(MANIFESTS_ROOT / "manifest.csv")
        self.test_df = manifest[manifest["split"] == "test"].copy()

        if self.pilot_mode:
            pilot_keys = []
            for (cat, vid), group in self.test_df.groupby(["category", "video_id"]):
                pilot_keys.append((cat, vid))
            pilot_keys = pilot_keys[:8]
            key_set = set(pilot_keys)
            self.test_df = self.test_df[
                self.test_df.apply(
                    lambda r: (str(r["category"]), str(r["video_id"])) in key_set,
                    axis=1,
                )
            ].copy()
            self.test_df = (
                self.test_df.groupby(["category", "video_id"], group_keys=False)
                .head(5)
                .copy()
            )

        self.sample_cols = ["category", "video_id", "original_frame_number"]
        self._validate_manifest_samples()

        logging.info(
            "Initialized %s seed %s: %d sample frames, %d categories, %d unique "
            "video IDs, %d conditions.",
            self.model_name,
            self.seed,
            len(self.test_df),
            self.test_df["category"].nunique(),
            self.test_df["video_id"].nunique(),
            len(self.conditions),
        )

    def _validate_manifest_samples(self) -> None:
        duplicated = self.test_df.duplicated(self.sample_cols, keep=False)
        if duplicated.any():
            rows = (
                self.test_df.loc[duplicated, self.sample_cols]
                .head(10)
                .to_dict("records")
            )
            raise RuntimeError(
                "Test manifest contains duplicate sample identities. "
                f"Examples: {rows}"
            )

    def _expected_keys(self) -> set:
        return {
            (
                str(r["category"]),
                str(r["video_id"]),
                int(r["original_frame_number"]),
            )
            for _, r in self.test_df.iterrows()
        }

    def _prepare_resume_file(self, frame_csv_path: Path) -> set:
        """
        Return exactly completed conditions.

        Any partial condition is removed before resuming. This prevents a
        restarted run from silently appending duplicate evaluation rows.
        """
        if not frame_csv_path.exists():
            return set()

        df = pd.read_csv(frame_csv_path, low_memory=False)
        required = {
            "category",
            "video_id",
            "original_frame_number",
            "transformation",
        }
        if not required.issubset(df.columns):
            logging.warning(
                "Existing frame CSV has an incompatible schema. Rebuilding it."
            )
            frame_csv_path.unlink()
            return set()

        expected = self._expected_keys()
        completed = set()
        keep_conditions = []

        for cond, group in df.groupby("transformation", sort=False):
            keys = {
                (
                    str(r["category"]),
                    str(r["video_id"]),
                    int(r["original_frame_number"]),
                )
                for _, r in group.iterrows()
            }
            has_duplicates = group.duplicated(
                ["category", "video_id", "original_frame_number"], keep=False
            ).any()

            if keys == expected and not has_duplicates and len(group) == len(expected):
                completed.add(str(cond))
                keep_conditions.append(str(cond))
            else:
                logging.warning(
                    "Removing incomplete/duplicate condition '%s' from resume CSV "
                    "(rows=%d, unique_keys=%d, expected=%d).",
                    cond,
                    len(group),
                    len(keys),
                    len(expected),
                )

        if len(keep_conditions) != df["transformation"].nunique():
            if keep_conditions:
                df = df[df["transformation"].astype(str).isin(keep_conditions)].copy()
                df.to_csv(frame_csv_path, index=False)
            else:
                frame_csv_path.unlink()

        return completed

    @staticmethod
    def _fieldnames() -> List[str]:
        return [
            "model",
            "seed",
            "video_id",
            "original_frame_number",
            "frame_path",
            "category",
            "ground_truth",
            "transformation",
            "transformation_family",
            "severity",
            "prob_fake",
            "pred_fake",
            "correct",
            "saliency_mass",
            "hit_rate",
            "saliency_entropy",
            "SO@10",
            "IoU@10",
            "SO@20",
            "IoU@20",
            "SO@30",
            "IoU@30",
            "SO",
            "IoU",
            "prob_fake_blur",
            "faithfulness_blur",
            "prob_fake_zero",
            "faithfulness_zero",
            "prob_fake_mean",
            "faithfulness_mean",
            "faithfulness",
            "clean_prob_fake",
            "clean_pred_fake",
            "prediction_preserved",
            "ES_cos",
            "explanation_IoU@10",
            "explanation_IoU@20",
            "explanation_IoU@30",
            "explanation_IoU",
            "prediction_state",
        ]

    def _validate_final_frame_csv(self, frame_csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(frame_csv_path, low_memory=False)
        expected = self._expected_keys()

        eval_cols = [
            "category",
            "video_id",
            "original_frame_number",
            "transformation",
        ]
        duplicates = df.duplicated(eval_cols, keep=False)
        if duplicates.any():
            examples = df.loc[duplicates, eval_cols].head(10).to_dict("records")
            raise RuntimeError(
                "Duplicate evaluation keys found after processing. "
                f"Examples: {examples}"
            )

        expected_conditions = {condition_tag(c) for c in self.conditions}
        actual_conditions = set(df["transformation"].astype(str).unique())

        if actual_conditions != expected_conditions:
            raise RuntimeError(
                "Condition coverage mismatch. "
                f"Expected={sorted(expected_conditions)}, "
                f"actual={sorted(actual_conditions)}"
            )

        for cond in sorted(expected_conditions):
            group = df[df["transformation"].astype(str) == cond]
            keys = {
                (
                    str(r["category"]),
                    str(r["video_id"]),
                    int(r["original_frame_number"]),
                )
                for _, r in group.iterrows()
            }
            if keys != expected or len(group) != len(expected):
                raise RuntimeError(
                    f"Condition '{cond}' has invalid sample coverage: "
                    f"rows={len(group)}, unique_keys={len(keys)}, "
                    f"expected={len(expected)}"
                )

        return df

    def run_evaluation(self) -> Dict[str, Any]:
        frame_csv_path = self.output_dir / "frame_level_results.csv"
        completed = self._prepare_resume_file(frame_csv_path)
        fields = self._fieldnames()
        write_header = not frame_csv_path.exists()

        clean_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

        if "clean" in completed:
            existing = pd.read_csv(frame_csv_path, low_memory=False)
            clean_rows = existing[existing["transformation"].astype(str) == "clean"]
            for _, r in clean_rows.iterrows():
                key = (
                    str(r["category"]),
                    str(r["video_id"]),
                    int(r["original_frame_number"]),
                )
                clean_cache[key] = {
                    "pred_fake": int(r["pred_fake"]),
                    "prob_fake": float(r["prob_fake"]),
                }
            del existing, clean_rows
            gc.collect()

        with open(frame_csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()

            if "clean" not in completed:
                logging.info("Running clean baseline.")
                for _, row in tqdm(
                    self.test_df.iterrows(),
                    total=len(self.test_df),
                    desc="Clean",
                ):
                    rec = self._process_frame(
                        row,
                        ("clean", "clean", 0, None),
                        clean_ref=None,
                    )
                    writer.writerow(rec)
                    key = (
                        str(row["category"]),
                        str(row["video_id"]),
                        int(row["original_frame_number"]),
                    )
                    clean_cache[key] = {
                        "pred_fake": int(rec["pred_fake"]),
                        "prob_fake": float(rec["prob_fake"]),
                    }
                fh.flush()
            else:
                logging.info("Clean condition already complete.")

            for cond_tuple in self.conditions:
                tag = condition_tag(cond_tuple)
                if tag == "clean" or tag in completed:
                    continue

                logging.info("Running condition %s.", tag)
                for _, row in tqdm(
                    self.test_df.iterrows(),
                    total=len(self.test_df),
                    desc=tag,
                ):
                    key = (
                        str(row["category"]),
                        str(row["video_id"]),
                        int(row["original_frame_number"]),
                    )
                    rec = self._process_frame(
                        row,
                        cond_tuple,
                        clean_ref=clean_cache.get(key),
                    )
                    writer.writerow(rec)
                fh.flush()
                gc.collect()

        clean_cache.clear()
        gc.collect()

        frame_df = self._validate_final_frame_csv(frame_csv_path)
        frame_df.to_csv(frame_csv_path, index=False)

        video_df = self._aggregate_to_video_level(frame_df)
        video_csv_path = self.output_dir / "video_level_results.csv"
        video_df.to_csv(video_csv_path, index=False)

        stats_df = self._run_statistical_analysis(video_df)
        stats_csv_path = self.output_dir / "statistics_results.csv"
        stats_df.to_csv(stats_csv_path, index=False)

        summary = {
            "model": self.model_name,
            "seed": self.seed,
            "pilot_mode": self.pilot_mode,
            "sample_identity": [
                "category",
                "video_id",
                "original_frame_number",
            ],
            "evaluation_identity": [
                "category",
                "video_id",
                "original_frame_number",
                "transformation",
            ],
            "total_frame_samples": int(len(self.test_df)),
            "total_frame_condition_evaluations": int(
                len(self.test_df) * len(self.conditions)
            ),
            "total_category_video_units": int(
                video_df[["category", "video_id"]].drop_duplicates().shape[0]
            ),
            "conditions_evaluated": [condition_tag(c) for c in self.conditions],
            "checkpoint_path": self.run_info["checkpoint_path"],
            "target_layers": [str(x) for x in self.generator.target_layers],
            "explanation_stability_threshold": EXPLANATION_STABILITY_THRESHOLD,
            "statistical_inference_unit": "source video ID after averaging across manipulation categories",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        try:
            from explainability_visuals import (
                generate_global_heatmaps,
                generate_representative_cases,
            )

            figures_dir = self.output_dir / "figures"
            generate_global_heatmaps(
                frame_df,
                self.cache_dir,
                figures_dir,
            )
            generate_representative_cases(
                frame_df,
                self.cache_dir,
                figures_dir,
            )
        except Exception as exc:
            logging.warning("Visual generation failed: %s", exc)

        self.mask_loader.close()
        return summary

    def _process_frame(
        self,
        row: pd.Series,
        cond_tuple: Tuple[str, str, int, Any],
        clean_ref: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cond_name, family, severity, param = cond_tuple
        tag = condition_tag(cond_tuple)

        video_id = str(row["video_id"])
        frame_number = int(row["original_frame_number"])
        category = str(row["category"])
        gt_label = 1 if str(row["label"]) == "fake" else 0

        img_bgr = cv2.imread(str(row["frame_path"]))
        if img_bgr is None:
            raise RuntimeError(f"Could not read frame: {row['frame_path']}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        transformed_rgb = apply_transformation_by_condition(image_rgb, cond_tuple)
        prob_fake, pred_fake = self.generator.predict(transformed_rgb)

        record: Dict[str, Any] = {
            "model": self.model_name,
            "seed": self.seed,
            "video_id": video_id,
            "original_frame_number": frame_number,
            "frame_path": str(row["frame_path"]),
            "category": category,
            "ground_truth": gt_label,
            "transformation": tag,
            "transformation_family": family,
            "severity": severity,
            "prob_fake": prob_fake,
            "pred_fake": pred_fake,
            "correct": int(pred_fake == gt_label),
        }

        if gt_label == 1:
            eval_h, eval_w = orig_h, orig_w
            eval_img = transformed_rgb

            if max(orig_h, orig_w) > 512:
                scale = 512.0 / max(orig_h, orig_w)
                eval_w = max(1, int(round(orig_w * scale)))
                eval_h = max(1, int(round(orig_h * scale)))
                eval_img = cv2.resize(
                    transformed_rgb,
                    (eval_w, eval_h),
                    interpolation=cv2.INTER_AREA,
                )

            cache_file = (
                self.cache_dir
                / tag
                / f"{category}_{video_id}_frame{frame_number:04d}.npy"
            )

            cam_map = self.generator.get_or_generate_cam(
                eval_img,
                cache_path=cache_file,
                target_shape=(eval_h, eval_w),
            )

            gt_mask = self.mask_loader.load_face_crop_mask(
                category=category,
                video_id=video_id,
                original_frame_number=frame_number,
                target_shape=(eval_h, eval_w),
            )
            if gt_mask is not None:
                record.update(evaluate_frame_localization(cam_map, gt_mask))

            salient_mask = get_salient_mask(
                cam_map,
                top_fraction=PRIMARY_SALIENCY_THRESHOLD,
            )

            # Faithfulness intervention is always applied to the original
            # transformed crop. This avoids mixing the CAM downsampling with
            # the model's baseline probability.
            mask_orig = cv2.resize(
                salient_mask.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            for method in FAITHFULNESS_METHODS:
                masked = mask_salient_region(
                    transformed_rgb,
                    mask_orig,
                    method=method,
                )
                masked_prob, _ = self.generator.predict(masked)
                record[f"prob_fake_{method}"] = masked_prob
                record[f"faithfulness_{method}"] = float(prob_fake) - float(masked_prob)
                del masked

            record["faithfulness"] = record.get(
                f"faithfulness_{PRIMARY_FAITHFULNESS_METHOD}",
                np.nan,
            )

            if clean_ref is not None:
                clean_cache_file = (
                    self.cache_dir
                    / "clean"
                    / f"{category}_{video_id}_frame{frame_number:04d}.npy"
                )
                if clean_cache_file.exists():
                    clean_cam = np.load(clean_cache_file).astype(np.float32)
                    if clean_cam.shape != cam_map.shape:
                        clean_cam = cv2.resize(
                            clean_cam,
                            (eval_w, eval_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    record.update(evaluate_frame_stability(clean_cam, cam_map))
                    del clean_cam

        if clean_ref is not None:
            clean_pred = int(clean_ref["pred_fake"])
            record["clean_prob_fake"] = float(clean_ref["prob_fake"])
            record["clean_pred_fake"] = clean_pred
            record["prediction_preserved"] = int(clean_pred == int(pred_fake))

            if gt_label == 1:
                if clean_pred == 1 and pred_fake == 1:
                    record["prediction_state"] = "Correct->Correct"
                elif clean_pred == 1 and pred_fake == 0:
                    record["prediction_state"] = "Correct->Incorrect"
                elif clean_pred == 0 and pred_fake == 1:
                    record["prediction_state"] = "Incorrect->Correct"
                else:
                    record["prediction_state"] = "Incorrect->Incorrect"

        return record

    def _aggregate_to_video_level(
        self,
        frame_df: pd.DataFrame,
    ) -> pd.DataFrame:
        group_cols = [
            "model",
            "seed",
            "category",
            "video_id",
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

        agg = {c: "mean" for c in metric_cols if c in frame_df.columns}
        agg["original_frame_number"] = "count"

        video_df = (
            frame_df.groupby(group_cols, as_index=False)
            .agg(agg)
            .rename(columns={"original_frame_number": "n_valid_frames"})
        )
        video_df["n_failed_frames"] = 0
        return video_df

    @staticmethod
    def _source_video_level(
        fake_video_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Collapse category-level fake-video rows to source-video rows.

        The same source video ID occurs under multiple FF++ manipulation
        categories. Averaging over category rows prevents those repeated
        source identities from becoming four independent observations.
        """
        numeric_candidates = [
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
        present = [c for c in numeric_candidates if c in fake_video_df.columns]
        return fake_video_df.groupby(
            ["model", "seed", "video_id", "transformation"],
            as_index=False,
        )[present].mean()

    def _paired_source_rows(
        self,
        source_df: pd.DataFrame,
        condition: str,
    ):
        clean = source_df[source_df["transformation"] == "clean"].set_index("video_id")
        transformed = source_df[source_df["transformation"] == condition].set_index(
            "video_id"
        )
        common = clean.index.intersection(transformed.index)
        return clean.loc[common], transformed.loc[common]

    def _run_statistical_analysis(
        self,
        video_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        fake_video = video_df[video_df["ground_truth"] == 1].copy()
        source_df = self._source_video_level(fake_video)

        conditions = [c for c in source_df["transformation"].unique() if c != "clean"]

        for cond in conditions:
            clean, transformed = self._paired_source_rows(source_df, cond)
            n = len(clean)
            if n == 0:
                continue

            # Family 1: localization
            for metric in [
                "SO",
                "IoU",
                "saliency_mass",
                "hit_rate",
            ]:
                if metric not in clean or metric not in transformed:
                    continue
                result = run_paired_wilcoxon_test(
                    clean[metric].to_numpy(dtype=float),
                    transformed[metric].to_numpy(dtype=float),
                )
                result.update(
                    {
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": metric,
                        "test_family": "family_1_localization",
                        "inference_unit": "source_video",
                    }
                )
                rows.append(result)

            # Family 2: faithfulness
            for method in [
                "faithfulness_blur",
                "faithfulness_zero",
                "faithfulness_mean",
            ]:
                if method not in clean or method not in transformed:
                    continue
                result = run_paired_wilcoxon_test(
                    clean[method].to_numpy(dtype=float),
                    transformed[method].to_numpy(dtype=float),
                )
                result.update(
                    {
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": method,
                        "test_family": "family_2_faithfulness",
                        "inference_unit": "source_video",
                    }
                )
                rows.append(result)

            # Family 3: explanation stability
            for metric in ["ES_cos", "explanation_IoU"]:
                if metric not in clean or metric not in transformed:
                    continue
                result = run_paired_wilcoxon_test(
                    clean[metric].to_numpy(dtype=float),
                    transformed[metric].to_numpy(dtype=float),
                )
                result.update(
                    {
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": metric,
                        "test_family": "family_3_stability",
                        "inference_unit": "source_video",
                    }
                )
                rows.append(result)

            # Family 4a: clean localization versus clean faithfulness.
            for loc, loc_name in [
                ("SO", "SO"),
                ("IoU", "IoU"),
                ("saliency_mass", "SM"),
            ]:
                for faith, faith_name in [
                    ("faithfulness_blur", "blur"),
                    ("faithfulness_zero", "zero"),
                    ("faithfulness_mean", "mean"),
                ]:
                    if loc not in clean or faith not in clean:
                        continue
                    x = clean[loc].to_numpy(dtype=float)
                    y = clean[faith].to_numpy(dtype=float)
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.sum() < MIN_SAMPLE_SPEARMAN:
                        rho, p = np.nan, np.nan
                        status = "not_tested_small_n"
                    else:
                        rho, p = spearmanr(x[valid], y[valid])
                        status = "tested"
                    rows.append(
                        {
                            "model": self.model_name,
                            "seed": self.seed,
                            "transformation": cond,
                            "metric": f"{loc_name}_vs_{faith_name}",
                            "test_family": "family_4a_loc_faith",
                            "inference_unit": "source_video",
                            "n_valid": int(valid.sum()),
                            "rho": float(rho) if np.isfinite(rho) else np.nan,
                            "raw_p": float(p) if np.isfinite(p) else np.nan,
                            "adjusted_p": np.nan,
                            "r_rb": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "test_status": status,
                        }
                    )

            # Family 4b: confidence shift versus explanation degradation.
            # ΔP_fake is a per-video confidence shift. It is not F1 loss.
            d_prob = (clean["prob_fake"] - transformed["prob_fake"]).to_numpy(
                dtype=float
            )

            for exp_metric, exp_name in [
                ("SO", "DSO"),
                ("IoU", "DIoU"),
                ("ES_cos", "Dstability"),
            ]:
                if exp_metric not in clean or exp_metric not in transformed:
                    continue

                d_exp = (clean[exp_metric] - transformed[exp_metric]).to_numpy(
                    dtype=float
                )
                valid = np.isfinite(d_prob) & np.isfinite(d_exp)

                if valid.sum() < MIN_SAMPLE_SPEARMAN:
                    rho, p = np.nan, np.nan
                    status = "not_tested_small_n"
                else:
                    rho, p = spearmanr(
                        d_prob[valid],
                        d_exp[valid],
                    )
                    status = "tested"

                rows.append(
                    {
                        "model": self.model_name,
                        "seed": self.seed,
                        "transformation": cond,
                        "metric": f"Dprob_vs_{exp_name}",
                        "test_family": "family_4b_det_exp_deg",
                        "inference_unit": "source_video",
                        "n_valid": int(valid.sum()),
                        "rho": float(rho) if np.isfinite(rho) else np.nan,
                        "raw_p": float(p) if np.isfinite(p) else np.nan,
                        "adjusted_p": np.nan,
                        "r_rb": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "test_status": status,
                    }
                )

        stats_df = pd.DataFrame(rows)
        if not stats_df.empty:
            stats_df = adjust_p_values_by_family(stats_df)
        return stats_df


def main():
    parser = argparse.ArgumentParser(description="Approach 3 explainability evaluation")
    parser.add_argument(
        "--model",
        choices=EXPLAINABILITY_MODELS,
        default="xception",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device, device_info = get_device()
    logging.info(
        "Starting Approach 3 on %s (%s).",
        device,
        device_info["device_name"],
    )

    models = EXPLAINABILITY_MODELS if args.check_all else [args.model]
    if args.check_all and args.seed is not None:
        raise ValueError("--seed cannot be combined with --check-all.")

    for model in models:
        logging.info("Processing %s.", model)
        orchestrator = ExplainabilityOrchestrator(
            model_name=model,
            seed=args.seed,
            device=device,
            pilot_mode=args.pilot,
        )
        orchestrator.run_evaluation()

    logging.info("Approach 3 evaluation completed.")


if __name__ == "__main__":
    main()
