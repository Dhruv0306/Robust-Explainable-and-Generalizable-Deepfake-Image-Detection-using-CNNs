"""
Unit tests for Phase 2: Robustness Configuration and Parameter Validation.

Validates:
1. Frozen severity anchors match Approach 2 specification (Plan Sections 10 & 18).
2. Core transformations default to enabled; optional default to disabled.
3. Condition enumeration yields exactly 13 conditions per checkpoint (117 total).
4. Strict boundary, type, and monotonicity validation catches invalid parameters.
5. Configuration serialization produces complete JSON and TXT experiment artifacts.
"""
import unittest
import json
import tempfile
import shutil
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from robustness_config import (
    DEFAULT_ROBUSTNESS_CONFIG,
    get_default_robustness_config,
    validate_robustness_config,
    get_enabled_conditions,
    save_robustness_config,
)
from config import ROBUSTNESS_ROOT, ROBUSTNESS_CONFIG, get_robustness_run_name


class TestPhase2RobustnessConfig(unittest.TestCase):
    """Test suite for Phase 2 robustness configuration."""

    def setUp(self):
        self.config = get_default_robustness_config()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_frozen_severity_anchors(self):
        """Verify frozen values match Plan Section 10 exactly."""
        # Core enabled flags
        self.assertTrue(self.config["jpeg"]["enabled"])
        self.assertTrue(self.config["resize"]["enabled"])
        self.assertTrue(self.config["brightness"]["enabled"])

        # Optional disabled flags
        self.assertFalse(self.config["gaussian_noise"]["enabled"])
        self.assertFalse(self.config["gaussian_blur"]["enabled"])
        self.assertFalse(self.config["contrast"]["enabled"])
        self.assertFalse(self.config["crop"]["enabled"])

        # Frozen numerical values
        self.assertEqual(self.config["jpeg"]["qualities"], [80, 50, 20])
        self.assertEqual(self.config["resize"]["scales"], [0.75, 0.50, 0.25])
        self.assertEqual(self.config["brightness"]["darker"], [0.80, 0.60, 0.40])
        self.assertEqual(self.config["brightness"]["brighter"], [1.20, 1.40, 1.60])

    def test_default_condition_enumeration(self):
        """Verify core configuration produces 13 conditions per checkpoint (1 clean + 12 corrupted)."""
        conditions = get_enabled_conditions(self.config)
        self.assertEqual(len(conditions), 13, f"Expected 13 conditions, got {len(conditions)}")

        # Condition 0 must be clean reference
        clean = conditions[0]
        self.assertEqual(clean["transformation"], "clean")
        self.assertEqual(clean["severity"], 0)
        self.assertIsNone(clean["parameter_value"])

        # Total matrix count across 9 checkpoints = 117
        total_evals = len(conditions) * 9
        self.assertEqual(total_evals, 117, f"Expected 117 evaluations across 9 checkpoints, got {total_evals}")

        # Check that JPEG, resize, dark, and bright conditions are all present
        transforms = [c["transformation"] for c in conditions[1:]]
        self.assertEqual(transforms.count("jpeg"), 3)
        self.assertEqual(transforms.count("resize"), 3)
        self.assertEqual(transforms.count("brightness_dark"), 3)
        self.assertEqual(transforms.count("brightness_bright"), 3)

    def test_optional_transformations_enumeration(self):
        """Verify optional transformations generate conditions only when explicitly enabled."""
        cfg = get_default_robustness_config()
        cfg["gaussian_noise"]["enabled"] = True
        cfg["contrast"]["enabled"] = True

        conditions = get_enabled_conditions(cfg)
        # 13 default + 3 noise + 3 contrast = 19
        self.assertEqual(len(conditions), 19)

        transforms = [c["transformation"] for c in conditions]
        self.assertIn("gaussian_noise", transforms)
        self.assertIn("contrast", transforms)
        self.assertNotIn("gaussian_blur", transforms)
        self.assertNotIn("crop", transforms)

    def test_jpeg_parameter_validation(self):
        """Verify JPEG validation rules."""
        # Non-decreasing qualities
        bad_cfg = get_default_robustness_config()
        bad_cfg["jpeg"]["qualities"] = [20, 50, 80]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Quality out of bounds
        bad_cfg["jpeg"]["qualities"] = [110, 50, 20]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Wrong list length
        bad_cfg["jpeg"]["qualities"] = [80, 50]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

    def test_resize_parameter_validation(self):
        """Verify resize validation rules."""
        bad_cfg = get_default_robustness_config()
        # Scale > 1.0
        bad_cfg["resize"]["scales"] = [1.2, 0.5, 0.25]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Scale <= 0.0
        bad_cfg["resize"]["scales"] = [0.75, 0.5, 0.0]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Non-decreasing scales
        bad_cfg["resize"]["scales"] = [0.25, 0.50, 0.75]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

    def test_brightness_parameter_validation(self):
        """Verify brightness validation rules."""
        bad_cfg = get_default_robustness_config()
        # Darkening factor > 1.0
        bad_cfg["brightness"]["darker"] = [1.1, 0.6, 0.4]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Brightening factor <= 1.0
        bad_cfg["brightness"]["brighter"] = [1.0, 1.4, 1.6]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

        # Non-increasing brightening factors
        bad_cfg["brightness"]["brighter"] = [1.6, 1.4, 1.2]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

    def test_blur_kernel_size_validation(self):
        """Verify Gaussian blur kernel size must be odd integer >= 3."""
        bad_cfg = get_default_robustness_config()
        bad_cfg["gaussian_blur"]["enabled"] = True
        # Even kernel size
        bad_cfg["gaussian_blur"]["kernel_sizes"] = [4, 6, 8]
        with self.assertRaises(ValueError):
            validate_robustness_config(bad_cfg)

    def test_config_serialization(self):
        """Verify serialization saves valid config.json and config.txt."""
        exp_id = "robustness_test_2026-09-10"
        json_path = save_robustness_config(
            self.config,
            self.temp_dir,
            experiment_id=exp_id,
            extra_metadata={"tester": "unit_test"},
        )
        self.assertTrue(json_path.exists())
        txt_path = self.temp_dir / "config.txt"
        self.assertTrue(txt_path.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_id"], exp_id)
        self.assertEqual(data["enabled_condition_count"], 13)
        self.assertIn("conditions", data)
        self.assertEqual(data["metadata"]["tester"], "unit_test")

        txt_content = txt_path.read_text(encoding="utf-8")
        self.assertIn(exp_id, txt_content)
        self.assertIn("Total Conditions per Checkpoint: 13", txt_content)


if __name__ == "__main__":
    unittest.main()
