"""
Unit tests for Approach 2 Novelty Extension Engine.

Validates:
1. Ingestion of 117-evaluation video predictions without missing records.
2. Novelty A schemas, class-stratified probability drift calculations, and flip metrics.
3. Novelty B Jaccard overlap bounds in [0.0, 1.0], correct empty-union handling (J=1.0), and 3-model consensus conservation.
4. Novelty C exact N=12 video accounting and per-manipulation Delta F1/Recall matrix dimensions.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import OUTPUT_ROOT
from robustness_novelty import (
    load_video_predictions,
    load_category_video_predictions,
    compute_confidence_decision_analysis,
    compute_pairwise_architecture_failure_agreement,
    compute_manipulation_vulnerability,
    compute_one_vs_original_metrics,
    audit_novelty_artifacts,
    run_novelty_pipeline,
)


class TestRobustnessNovelty(unittest.TestCase):
    """Test suite for Approach 2 novelty analytics."""

    @classmethod
    def setUpClass(cls):
        cls.exp_dir = OUTPUT_ROOT / "robustness" / "core_experiment_117"
        if not cls.exp_dir.exists():
            raise FileNotFoundError(f"core_experiment_117 missing at {cls.exp_dir}")

        cls.video_df = load_video_predictions(cls.exp_dir)
        cls.cat_video_df = load_category_video_predictions(cls.exp_dir)

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_phase0_artifact_audit(self):
        """Verify all 117 conditions, schemas, checkpoints, and category counts before analysis."""
        audit = audit_novelty_artifacts(self.exp_dir)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["master_rows"], 117)
        self.assertEqual(audit["video_prediction_rows"], 2808)
        self.assertEqual(audit["conditions_per_checkpoint"], 13)
        self.assertEqual(set(audit["clean_category_rows"]), {"Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"})
        self.assertTrue(all(v == 108 for v in audit["clean_category_rows"].values()))

    def test_ingestion_counts(self):
        """Verify video prediction ingestion captures exactly 9 checkpoints x 13 conditions x 24 videos = 2,808 rows."""
        self.assertEqual(len(self.video_df), 2808)
        self.assertEqual(self.video_df["model"].nunique(), 3)
        self.assertEqual(self.video_df["seed"].nunique(), 3)

        # Category predictions: exactly 12 videos per category
        for cat, grp in self.cat_video_df[self.cat_video_df["severity"] == 0].groupby("category"):
            # 9 checkpoints x 12 videos = 108
            self.assertEqual(len(grp), 108, f"Unexpected row count for category {cat}: {len(grp)}")

    def test_novelty_a_confidence_and_decision(self):
        """Verify class-stratified drift, flip counts, and error transition conservation."""
        conf_df, flip_df, error_df = compute_confidence_decision_analysis(self.video_df)

        self.assertEqual(len(conf_df), 117)
        self.assertEqual(len(flip_df), 117)
        self.assertEqual(len(error_df), 117)

        # Verify columns in confidence summary
        expected_conf_cols = {
            "overall_mean_delta_p", "real_mean_delta_p", "fake_mean_delta_p"
        }
        self.assertTrue(expected_conf_cols.issubset(set(conf_df.columns)))

        # Verify flip rates in [0, 1]
        self.assertTrue((flip_df["flip_rate"] >= 0.0).all())
        self.assertTrue((flip_df["flip_rate"] <= 1.0).all())

        # Clean condition must have 0 flips and 0 delta
        clean_flips = flip_df[flip_df["severity"] == 0]
        self.assertTrue((clean_flips["flip_rate"] == 0.0).all())
        self.assertTrue((clean_flips["flip_count"] == 0).all())

        clean_conf = conf_df[conf_df["severity"] == 0]
        self.assertTrue((clean_conf["overall_mean_delta_p"] == 0.0).all())
        self.assertTrue((clean_conf["overall_mean_abs_delta_p"] == 0.0).all())
        self.assertTrue((clean_conf["real_mean_delta_p"] == 0.0).all())
        self.assertTrue((clean_conf["fake_mean_delta_p"] == 0.0).all())

        # Verify error transitions sum to total videos (24) per condition
        error_df["total_check"] = (
            error_df["stable_correct"]
            + error_df["robustness_failure"]
            + error_df["transformation_correction"]
            + error_df["persistent_error"]
        )
        self.assertTrue((error_df["total_check"] == 24).all())

    def test_jaccard_empty_union_is_complete_agreement(self):
        """Verify the documented empty failure-union convention J=1.0."""
        from robustness_novelty import compute_pairwise_architecture_failure_agreement
        toy = pd.DataFrame([
            {"model": m, "seed": 42, "video_id": "v1", "label": 1, "pred_fake": 1, "prob_fake": 0.9, "transformation": "clean", "severity": 0, "direction": "none", "parameter_name": "none", "parameter_value": None}
            for m in ["efficientnet_b0", "resnet50", "xception"]
        ] + [
            {"model": m, "seed": 42, "video_id": "v1", "label": 1, "pred_fake": 1, "prob_fake": 0.9, "transformation": "jpeg", "severity": 1, "direction": "none", "parameter_name": "quality", "parameter_value": 80}
            for m in ["efficientnet_b0", "resnet50", "xception"]
        ])
        overlap, _, _ = compute_pairwise_architecture_failure_agreement(toy)
        self.assertTrue((overlap["jaccard_overlap"] == 1.0).all())

    def test_novelty_b_architecture_failure_agreement(self):
        """Verify Jaccard overlap bounds in [0.0, 1.0], empty-union J=1.0, and consensus."""
        overlap_df, disagree_df, consensus_df = compute_pairwise_architecture_failure_agreement(self.video_df)

        # 3 pairs x 3 seeds x 12 corrupted conditions = 108
        self.assertEqual(len(overlap_df), 108)
        self.assertEqual(len(disagree_df), 108)

        # Jaccard index bounds
        self.assertTrue((overlap_df["jaccard_overlap"] >= 0.0).all())
        self.assertTrue((overlap_df["jaccard_overlap"] <= 1.0).all())

        # Disagreement rate bounds
        self.assertTrue((disagree_df["disagreement_rate"] >= 0.0).all())
        self.assertTrue((disagree_df["disagreement_rate"] <= 1.0).all())

        # Consensus total videos sum to 24
        consensus_df["vid_sum"] = (
            consensus_df["models_failed_0"]
            + consensus_df["models_failed_1"]
            + consensus_df["models_failed_2"]
            + consensus_df["models_failed_3"]
        )
        self.assertTrue((consensus_df["vid_sum"] == 24).all())

    def test_clean_transformed_pairing(self):
        """Verify every transformed checkpoint condition has exactly 24 paired videos."""
        from robustness_novelty import compute_confidence_decision_analysis
        conf_df, flip_df, error_df = compute_confidence_decision_analysis(self.video_df)
        self.assertTrue((flip_df["total_videos"] == 24).all())
        self.assertTrue((error_df["total_evals"] == 24).all())

    def test_synthetic_f1_vs_recall_regression(self):
        """Regression test ensuring positive-class F1 correctly incorporates false positives."""
        fake_preds = pd.Series([1, 1, 0, 0])
        original_preds = pd.Series([0, 0, 0, 0])
        res_no_fp = compute_one_vs_original_metrics(fake_preds, original_preds)
        self.assertAlmostEqual(res_no_fp["recall"], 0.5, places=4)
        self.assertAlmostEqual(res_no_fp["f1"], 2 * 2 / (2 * 2 + 0 + 2), places=4)  # 4/6 = 0.6667
        self.assertNotEqual(res_no_fp["recall"], res_no_fp["f1"])

        # When FP > 0, F1 is strictly smaller than the no-FP case (4/7 < 4/6)
        original_preds_with_fp = pd.Series([1, 0, 0, 0])
        res_with_fp = compute_one_vs_original_metrics(fake_preds, original_preds_with_fp)
        self.assertAlmostEqual(res_with_fp["recall"], 0.5, places=4)
        self.assertAlmostEqual(res_with_fp["f1"], 2 * 2 / (2 * 2 + 1 + 2), places=4)  # 4/7 = 0.5714
        self.assertLess(res_with_fp["f1"], res_no_fp["f1"])

    def test_category_mapping_and_uniqueness(self):
        """
        Verify the intended mapping structure:
        - 12 Original videos -> 1 category record ('Original')
        - 12 Fake videos -> 4 category records (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
        - Total 60 unique (video_id, category) pairs
        - Composite prediction key is strictly unique per checkpoint condition
        """
        clean_cat = self.cat_video_df[self.cat_video_df["severity"] == 0]
        unique_pairs = clean_cat[["video_id", "category"]].drop_duplicates()
        self.assertEqual(len(unique_pairs), 60)

        # Check multiplicity per video_id
        vid_to_cats = unique_pairs.groupby("video_id")["category"].nunique()
        self.assertEqual((vid_to_cats == 1).sum(), 12)  # 12 real videos have 1 category
        self.assertEqual((vid_to_cats == 4).sum(), 12)  # 12 fake videos have 4 categories

        # Composite prediction key uniqueness across all 7,020 rows
        key_cols = ["model", "seed", "transformation", "severity", "direction", "parameter_name", "parameter_value", "video_id", "category"]
        self.assertFalse(self.cat_video_df.duplicated(subset=key_cols).any(), "Duplicate category prediction key detected")

    def test_novelty_c_manipulation_vulnerability(self):
        """Verify manipulation vulnerability calculations and exact N=12 accounting."""
        cat_agg_df, f1_piv, recall_piv = compute_manipulation_vulnerability(self.cat_video_df)

        # 4 fake categories as index
        expected_cats = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
        self.assertEqual(set(f1_piv.index), expected_cats)
        self.assertEqual(set(recall_piv.index), expected_cats)

        # 4 core transformations as columns
        expected_transforms = {"brightness_bright", "brightness_dark", "jpeg", "resize"}
        self.assertEqual(set(f1_piv.columns), expected_transforms)
        self.assertEqual(set(recall_piv.columns), expected_transforms)

        # Under severe JPEG Q=20, Delta F1 should be strongly negative (~ -0.80)
        self.assertTrue((f1_piv["jpeg"] < -0.7).all())

        # F1 must use false positives and therefore differ from recall for
        # at least one category-condition; this catches recall-as-F1 regressions.
        self.assertTrue((cat_agg_df["f1_mean"] - cat_agg_df["recall_mean"]).abs().max() > 0.0)

    def test_end_to_end_novelty_pipeline(self):
        """Verify full run_novelty_pipeline execution and file creation."""
        res = run_novelty_pipeline(self.exp_dir, output_dir=self.temp_dir)
        self.assertEqual(res["output_dir"], self.temp_dir)

        for table_name in res["saved_tables"]:
            p = self.temp_dir / table_name
            self.assertTrue(p.exists(), f"Missing expected output table: {table_name}")
            self.assertGreater(p.stat().st_size, 100)

        fig_dir = self.temp_dir / "figures"
        self.assertTrue((fig_dir / "manipulation_vulnerability_delta_f1_heatmap.png").exists())
        self.assertTrue((fig_dir / "architecture_failure_overlap_jaccard_heatmap.png").exists())
        self.assertTrue((fig_dir / "confidence_drift_and_flip_rate_analysis.png").exists())


if __name__ == "__main__":
    unittest.main()
