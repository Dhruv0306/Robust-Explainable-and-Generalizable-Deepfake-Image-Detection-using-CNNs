"""
Unit tests for Phase 3: Robustness Transformation Module.

Validates:
1. Shape preservation: (H, W, 3) -> (H, W, 3) for all transformations.
2. Dtype and value range: uint8 in [0, 255] with no integer overflow.
3. RGB/BGR channel safety: no silent blue/red channel swapping during JPEG or other ops.
4. Monotonic severity degradation for all 7 transformation types.
5. Deterministic random state handling for Gaussian noise.
6. Error handling and boundary validation for malformed inputs.
"""
import unittest
import numpy as np

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from robustness import (
    apply_jpeg,
    apply_resize,
    apply_brightness,
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_contrast,
    apply_crop,
    apply_robustness_transform,
    apply_condition,
    TransformationError,
)
from robustness_config import get_default_robustness_config, get_enabled_conditions


class TestPhase3RobustnessTransforms(unittest.TestCase):
    """Test suite for Phase 3 transformation implementations."""

    def setUp(self):
        # Create a synthetic image with distinct channels and spatial structure
        np.random.seed(42)
        self.h, self.w = 120, 120
        # Gradient pattern with strongly asymmetric channels (R >> G >> B)
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.img[:, :, 0] = np.linspace(150, 240, self.h)[:, None]  # R
        self.img[:, :, 1] = np.linspace(50, 120, self.w)[None, :]   # G
        self.img[:, :, 2] = 30                                      # B

    def test_shape_and_dtype_preservation(self):
        """Verify all transformations preserve (H, W, 3) shape and uint8 dtype."""
        transforms = [
            apply_jpeg(self.img, 50),
            apply_resize(self.img, 0.5),
            apply_brightness(self.img, 0.6),
            apply_brightness(self.img, 1.4),
            apply_gaussian_noise(self.img, 15.0, seed=42),
            apply_gaussian_blur(self.img, 5, 2.0),
            apply_contrast(self.img, 0.6),
            apply_crop(self.img, 0.8),
        ]
        for t in transforms:
            self.assertEqual(t.shape, self.img.shape)
            self.assertEqual(t.dtype, np.uint8)
            self.assertGreaterEqual(t.min(), 0)
            self.assertLessEqual(t.max(), 255)

    def test_jpeg_channel_safety_and_severity_ordering(self):
        """Verify JPEG does not invert R/B channels and artifact increases as quality drops."""
        # Check channel order: R channel should remain much greater than B
        jpg_80 = apply_jpeg(self.img, 80)
        jpg_50 = apply_jpeg(self.img, 50)
        jpg_20 = apply_jpeg(self.img, 20)

        self.assertGreater(jpg_80[:, :, 0].mean(), jpg_80[:, :, 2].mean() + 80)
        self.assertGreater(jpg_20[:, :, 0].mean(), jpg_20[:, :, 2].mean() + 80)

        # Higher compression (lower quality) should increase pixel-level deviation from clean
        diff_80 = np.abs(jpg_80.astype(float) - self.img.astype(float)).mean()
        diff_50 = np.abs(jpg_50.astype(float) - self.img.astype(float)).mean()
        diff_20 = np.abs(jpg_20.astype(float) - self.img.astype(float)).mean()

        self.assertLess(diff_80, diff_50)
        self.assertLess(diff_50, diff_20)

    def test_resize_severity_ordering(self):
        """Verify resolution loss increases monotonically as scale decreases on high-frequency patterns."""
        # Use high-frequency checkerboard pattern to measure spatial resolution loss
        x = np.arange(self.w)
        y = np.arange(self.h)
        xx, yy = np.meshgrid(x, y)
        pattern = (((xx // 4) % 2) ^ ((yy // 4) % 2)) * 200
        test_img = np.stack([pattern, pattern, pattern], axis=2).astype(np.uint8)

        res_75 = apply_resize(test_img, 0.75)
        res_50 = apply_resize(test_img, 0.50)
        res_25 = apply_resize(test_img, 0.25)

        diff_75 = np.abs(res_75.astype(float) - test_img.astype(float)).mean()
        diff_50 = np.abs(res_50.astype(float) - test_img.astype(float)).mean()
        diff_25 = np.abs(res_25.astype(float) - test_img.astype(float)).mean()

        self.assertLess(diff_75, diff_50)
        self.assertLess(diff_50, diff_25)

    def test_brightness_darkening_and_brightening(self):
        """Verify darkening reduces and brightening increases luminance monotonically."""
        clean_mean = self.img.mean()

        d_80 = apply_brightness(self.img, 0.80)
        d_60 = apply_brightness(self.img, 0.60)
        d_40 = apply_brightness(self.img, 0.40)

        # Monotonically darker
        self.assertGreater(clean_mean, d_80.mean())
        self.assertGreater(d_80.mean(), d_60.mean())
        self.assertGreater(d_60.mean(), d_40.mean())

        b_120 = apply_brightness(self.img, 1.20)
        b_140 = apply_brightness(self.img, 1.40)
        b_160 = apply_brightness(self.img, 1.60)

        # Monotonically brighter
        self.assertLess(clean_mean, b_120.mean())
        self.assertLess(b_120.mean(), b_140.mean())
        self.assertLess(b_140.mean(), b_160.mean())

    def test_gaussian_noise_determinism_and_severity(self):
        """Verify noise is deterministic for identical seeds and varies with seed/severity."""
        # Identical seed produces identical corrupted pixels
        n1 = apply_gaussian_noise(self.img, 15.0, seed=123)
        n2 = apply_gaussian_noise(self.img, 15.0, seed=123)
        self.assertTrue(np.array_equal(n1, n2))

        # Different seed produces different pixels
        n3 = apply_gaussian_noise(self.img, 15.0, seed=456)
        self.assertFalse(np.array_equal(n1, n3))

        # Monotonic noise magnitude
        diff_5 = np.abs(apply_gaussian_noise(self.img, 5.0, seed=42).astype(float) - self.img).mean()
        diff_15 = np.abs(apply_gaussian_noise(self.img, 15.0, seed=42).astype(float) - self.img).mean()
        diff_30 = np.abs(apply_gaussian_noise(self.img, 30.0, seed=42).astype(float) - self.img).mean()

        self.assertLess(diff_5, diff_15)
        self.assertLess(diff_15, diff_30)

    def test_gaussian_blur_severity(self):
        """Verify blur increases deviation from sharp edges as kernel/sigma grow."""
        b1 = apply_gaussian_blur(self.img, kernel_size=3, sigma=1.0)
        b2 = apply_gaussian_blur(self.img, kernel_size=5, sigma=2.0)
        b3 = apply_gaussian_blur(self.img, kernel_size=7, sigma=3.0)

        diff_1 = np.abs(b1.astype(float) - self.img.astype(float)).mean()
        diff_2 = np.abs(b2.astype(float) - self.img.astype(float)).mean()
        diff_3 = np.abs(b3.astype(float) - self.img.astype(float)).mean()

        self.assertLess(diff_1, diff_2)
        self.assertLess(diff_2, diff_3)

    def test_contrast_monotonicity(self):
        """Verify contrast factor scales standard deviation around mean."""
        std_clean = self.img.astype(float).std()
        c_80 = apply_contrast(self.img, 0.80)
        c_60 = apply_contrast(self.img, 0.60)
        c_40 = apply_contrast(self.img, 0.40)

        self.assertGreater(std_clean, c_80.astype(float).std())
        self.assertGreater(c_80.astype(float).std(), c_60.astype(float).std())
        self.assertGreater(c_60.astype(float).std(), c_40.astype(float).std())

    def test_centered_crop_geometry(self):
        """Verify centered crop removes peripheral information and scales with ratio."""
        c_90 = apply_crop(self.img, 0.90)
        c_70 = apply_crop(self.img, 0.70)

        diff_90 = np.abs(c_90.astype(float) - self.img.astype(float)).mean()
        diff_70 = np.abs(c_70.astype(float) - self.img.astype(float)).mean()

        self.assertLess(diff_90, diff_70)

    def test_apply_condition_dispatch(self):
        """Verify apply_condition successfully processes all enabled conditions."""
        cfg = get_default_robustness_config()
        conditions = get_enabled_conditions(cfg)

        for cond in conditions:
            transformed, meta = apply_condition(self.img, cond)
            self.assertEqual(transformed.shape, self.img.shape)
            self.assertEqual(transformed.dtype, np.uint8)
            self.assertTrue(meta["success"])
            self.assertEqual(meta["severity"], cond["severity"])
            self.assertEqual(meta["transformation"], cond["transformation"])

    def test_invalid_input_rejection(self):
        """Verify input validation rejects non-uint8 or invalid dimensional arrays."""
        float_img = self.img.astype(np.float32)
        with self.assertRaises(ValueError):
            apply_jpeg(float_img, 50)

        grayscale_img = self.img[:, :, 0]
        with self.assertRaises(ValueError):
            apply_resize(grayscale_img, 0.5)


if __name__ == "__main__":
    unittest.main()
