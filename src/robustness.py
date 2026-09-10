"""
Approach 2 Robustness Transformation Engine.

Applies controlled image corruptions to face crops before model-specific resizing:
1. JPEG compression (quantization via cv2 imencode/imdecode with explicit RGB/BGR safety)
2. Lower-resolution resizing (downsampling followed by upsampling to original crop resolution)
3. Brightness changes (darkening and brightening via float32 scaling and uint8 clipping)
4. Gaussian noise (additive Gaussian perturbation with deterministic seeding)
5. Gaussian blur (spatial low-pass filtering with monotonic kernels and sigmas)
6. Contrast changes (photometric contrast scaling around channel mean)
7. Controlled centered cropping (peripheral spatial information removal with original shape restoration)

All transformations operate on numpy uint8 arrays in RGB format and return valid
RGB uint8 arrays matching the input dimensions.
"""
import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class TransformationError(Exception):
    """Raised when an image corruption operation fails."""
    pass


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Apply JPEG compression and decompression at the specified quality level.

    Args:
        image: RGB uint8 array (H, W, 3).
        quality: JPEG quality factor in [1, 100].

    Returns:
        Transformed RGB uint8 array (H, W, 3).
    """
    if not isinstance(quality, int) or quality < 1 or quality > 100:
        raise ValueError(f"JPEG quality must be an integer in [1, 100], got {quality}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    # Explicit RGB -> BGR conversion before passing to OpenCV encoder
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_param)

    if not success or encoded is None or len(encoded) == 0:
        raise TransformationError(f"JPEG encoding failed for quality={quality}")

    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise TransformationError(f"JPEG decoding failed for quality={quality}")

    # Explicit BGR -> RGB conversion after decoding
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    if decoded_rgb.shape != image.shape:
        raise TransformationError(f"JPEG output shape mismatch: {decoded_rgb.shape} vs expected {image.shape}")

    return decoded_rgb


def apply_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Simulate spatial resolution loss by downsampling then upsampling back to original size.

    Args:
        image: RGB uint8 array (H, W, 3).
        scale: Downsampling scale factor in (0.0, 1.0].

    Returns:
        Transformed RGB uint8 array (H, W, 3) restored to original dimensions.
    """
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"Resize scale must be in (0.0, 1.0], got {scale}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    if scale == 1.0:
        return image.copy()

    h, w = image.shape[:2]
    low_w = max(1, int(round(w * scale)))
    low_h = max(1, int(round(h * scale)))

    downsampled = cv2.resize(image, (low_w, low_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Apply photometric brightness scaling using float32 arithmetic and clipping.

    Args:
        image: RGB uint8 array (H, W, 3).
        factor: Multiplicative factor > 0.0 (< 1.0 for darker, > 1.0 for brighter).

    Returns:
        Transformed RGB uint8 array (H, W, 3).
    """
    if factor <= 0.0:
        raise ValueError(f"Brightness factor must be > 0.0, got {factor}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    if factor == 1.0:
        return image.copy()

    # Photometric arithmetic safety: float32 multiplication followed by clipping
    scaled = image.astype(np.float32) * factor
    clipped = np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    return clipped


def apply_gaussian_noise(image: np.ndarray, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to image pixels with deterministic seed control.

    Args:
        image: RGB uint8 array (H, W, 3).
        sigma: Standard deviation of noise > 0.0.
        seed: Random seed for deterministic perturbation across model evaluations.

    Returns:
        Transformed RGB uint8 array (H, W, 3).
    """
    if sigma <= 0.0:
        raise ValueError(f"Gaussian noise sigma must be > 0.0, got {sigma}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)

    noisy = image.astype(np.float32) + noise
    clipped = np.clip(noisy, 0.0, 255.0).astype(np.uint8)

    return clipped


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply spatial Gaussian blur filter.

    Args:
        image: RGB uint8 array (H, W, 3).
        kernel_size: Odd integer >= 3.
        sigma: Gaussian kernel standard deviation > 0.0.

    Returns:
        Transformed RGB uint8 array (H, W, 3).
    """
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"Gaussian blur kernel_size must be an odd integer >= 3, got {kernel_size}")
    if sigma <= 0.0:
        raise ValueError(f"Gaussian blur sigma must be > 0.0, got {sigma}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale image contrast relative to per-channel mean in float32.

    Args:
        image: RGB uint8 array (H, W, 3).
        factor: Multiplicative contrast factor > 0.0.

    Returns:
        Transformed RGB uint8 array (H, W, 3).
    """
    if factor <= 0.0:
        raise ValueError(f"Contrast factor must be > 0.0, got {factor}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    if factor == 1.0:
        return image.copy()

    img_f = image.astype(np.float32)
    # Scale deviations from per-channel mean to prevent color shifting
    mean = np.mean(img_f, axis=(0, 1), keepdims=True)
    adjusted = (img_f - mean) * factor + mean
    clipped = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

    return clipped


def apply_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    """
    Apply centered cropping and resize back to original dimensions.

    Args:
        image: RGB uint8 array (H, W, 3).
        ratio: Center crop dimension ratio in (0.0, 1.0).

    Returns:
        Transformed RGB uint8 array (H, W, 3) restored to original dimensions.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"Crop ratio must be in (0.0, 1.0), got {ratio}")

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 image of shape (H, W, 3), got dtype={image.dtype}, shape={image.shape}")

    h, w = image.shape[:2]
    crop_h = max(1, int(round(h * ratio)))
    crop_w = max(1, int(round(w * ratio)))

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    cropped = image[top : top + crop_h, left : left + crop_w]
    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored


def apply_robustness_transform(
    image: np.ndarray,
    transformation_name: str,
    severity: int,
    parameters: Optional[Any] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified transformation dispatch interface.

    Args:
        image: Original face crop as RGB uint8 array (H, W, 3).
        transformation_name: One of 'clean', 'jpeg', 'resize', 'brightness_dark',
            'brightness_bright', 'gaussian_noise', 'gaussian_blur', 'contrast', 'crop'.
        severity: Severity level (0 for clean, 1..3 for corruptions).
        parameters: Optional parameter value (e.g. quality, scale factor, etc.).
        seed: Optional seed for stochastic transformations (Gaussian noise).

    Returns:
        (transformed_image, metadata_dict)
    """
    meta: Dict[str, Any] = {
        "transformation": transformation_name,
        "severity": severity,
        "parameter_value": parameters,
        "success": True,
        "error_message": None,
    }

    if transformation_name == "clean" or severity == 0:
        return image.copy(), meta

    try:
        if transformation_name == "jpeg":
            transformed = apply_jpeg(image, quality=int(parameters))
        elif transformation_name == "resize":
            transformed = apply_resize(image, scale=float(parameters))
        elif transformation_name in ("brightness_dark", "brightness_bright", "brightness"):
            transformed = apply_brightness(image, factor=float(parameters))
        elif transformation_name == "gaussian_noise":
            transformed = apply_gaussian_noise(image, sigma=float(parameters), seed=seed)
        elif transformation_name == "gaussian_blur":
            if isinstance(parameters, dict):
                k = parameters["kernel_size"]
                s = parameters["sigma"]
            else:
                raise ValueError("Gaussian blur parameters must be dict with 'kernel_size' and 'sigma'")
            transformed = apply_gaussian_blur(image, kernel_size=k, sigma=s)
        elif transformation_name == "contrast":
            transformed = apply_contrast(image, factor=float(parameters))
        elif transformation_name == "crop":
            transformed = apply_crop(image, ratio=float(parameters))
        else:
            raise ValueError(f"Unknown transformation name: {transformation_name}")

        return transformed, meta

    except Exception as e:
        meta["success"] = False
        meta["error_message"] = str(e)
        logging.error(f"Transformation failure [{transformation_name}, sev={severity}]: {e}")
        raise TransformationError(f"Failed to apply {transformation_name} (sev={severity}): {e}") from e


def apply_condition(
    image: np.ndarray,
    condition: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper to apply a condition dict from get_enabled_conditions().
    """
    t_name = condition["transformation"]
    sev = condition["severity"]
    param = condition.get("parameter_value")
    # Condition-level seed takes precedence if defined (e.g. gaussian_noise)
    cond_seed = condition.get("seed", seed)

    transformed, meta = apply_robustness_transform(
        image=image,
        transformation_name=t_name,
        severity=sev,
        parameters=param,
        seed=cond_seed,
    )
    meta["condition_id"] = condition.get("condition_id", f"{t_name}_sev{sev}")
    meta["direction"] = condition.get("direction", "none")
    return transformed, meta
