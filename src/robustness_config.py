"""
Approach 2 Robustness Experiment Configuration and Parameter Validation.

Freezes numerical severity anchors for core transformations (JPEG, Resize, Brightness)
and optional transformations (Gaussian Noise, Gaussian Blur, Contrast, Centered Crop).
Provides schema validation, condition enumeration, and configuration serialization.
"""
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Frozen core severity parameters and defaults as specified in Approach 2 Plan Section 10 & 18
DEFAULT_ROBUSTNESS_CONFIG: Dict[str, Any] = {
    "jpeg": {
        "enabled": True,
        "qualities": [80, 50, 20],  # Quality levels for severities 1, 2, 3
    },
    "resize": {
        "enabled": True,
        "scales": [0.75, 0.50, 0.25],  # Resolution scales for severities 1, 2, 3
    },
    "brightness": {
        "enabled": True,
        "darker": [0.80, 0.60, 0.40],  # Photometric factors for severities 1, 2, 3
        "brighter": [1.20, 1.40, 1.60],  # Photometric factors for severities 1, 2, 3
    },
    "gaussian_noise": {
        "enabled": False,
        "sigmas": [5.0, 15.0, 30.0],  # Noise standard deviations
        "seed": 42,
    },
    "gaussian_blur": {
        "enabled": False,
        "kernel_sizes": [3, 5, 7],  # Must be positive odd integers
        "sigmas": [1.0, 2.0, 3.0],
    },
    "contrast": {
        "enabled": False,
        "factors": [0.80, 0.60, 0.40],  # Contrast factors
    },
    "crop": {
        "enabled": False,
        "ratios": [0.90, 0.80, 0.70],  # Centered crop dimension ratios
    },
}


def get_default_robustness_config() -> Dict[str, Any]:
    """Return a deep copy of the default frozen robustness configuration."""
    return copy.deepcopy(DEFAULT_ROBUSTNESS_CONFIG)


def validate_robustness_config(config: Dict[str, Any]) -> None:
    """
    Validate robustness configuration parameters against plan specifications.

    Raises:
        ValueError: If any parameter violates monotonicity, bounds, or types.
    """
    if not isinstance(config, dict):
        raise ValueError(f"Robustness config must be a dict, got {type(config).__name__}")

    # 1. JPEG Compression
    if "jpeg" not in config:
        raise ValueError("Missing 'jpeg' in robustness config")
    jpeg = config["jpeg"]
    if not isinstance(jpeg.get("enabled"), bool):
        raise ValueError("jpeg.enabled must be a boolean")
    qualities = jpeg.get("qualities")
    if not isinstance(qualities, list) or len(qualities) != 3:
        raise ValueError("jpeg.qualities must be a list of exactly 3 integers")
    for q in qualities:
        if not isinstance(q, int) or q < 1 or q > 100:
            raise ValueError(f"JPEG quality must be an integer in [1, 100], got {q}")
    if not (qualities[0] > qualities[1] > qualities[2]):
        raise ValueError(f"JPEG qualities must strictly decrease with severity: {qualities}")

    # 2. Lower-resolution Resizing
    if "resize" not in config:
        raise ValueError("Missing 'resize' in robustness config")
    resize = config["resize"]
    if not isinstance(resize.get("enabled"), bool):
        raise ValueError("resize.enabled must be a boolean")
    scales = resize.get("scales")
    if not isinstance(scales, list) or len(scales) != 3:
        raise ValueError("resize.scales must be a list of exactly 3 floats")
    for s in scales:
        if not isinstance(s, (int, float)) or s <= 0.0 or s > 1.0:
            raise ValueError(f"Resize scale must be in (0.0, 1.0], got {s}")
    if not (scales[0] > scales[1] > scales[2]):
        raise ValueError(f"Resize scales must strictly decrease with severity: {scales}")

    # 3. Brightness Changes
    if "brightness" not in config:
        raise ValueError("Missing 'brightness' in robustness config")
    brightness = config["brightness"]
    if not isinstance(brightness.get("enabled"), bool):
        raise ValueError("brightness.enabled must be a boolean")
    darker = brightness.get("darker")
    if not isinstance(darker, list) or len(darker) != 3:
        raise ValueError("brightness.darker must be a list of exactly 3 floats")
    for d in darker:
        if not isinstance(d, (int, float)) or d <= 0.0 or d > 1.0:
            raise ValueError(f"Brightness darker factor must be in (0.0, 1.0], got {d}")
    if not (darker[0] > darker[1] > darker[2]):
        raise ValueError(f"Darkening factors must strictly decrease with severity: {darker}")

    brighter = brightness.get("brighter")
    if not isinstance(brighter, list) or len(brighter) != 3:
        raise ValueError("brightness.brighter must be a list of exactly 3 floats")
    for b in brighter:
        if not isinstance(b, (int, float)) or b <= 1.0:
            raise ValueError(f"Brightness brighter factor must be > 1.0, got {b}")
    if not (brighter[0] < brighter[1] < brighter[2]):
        raise ValueError(f"Brightening factors must strictly increase with severity: {brighter}")

    # 4. Gaussian Noise (Optional)
    if "gaussian_noise" in config:
        gnoise = config["gaussian_noise"]
        if not isinstance(gnoise.get("enabled"), bool):
            raise ValueError("gaussian_noise.enabled must be a boolean")
        sigmas = gnoise.get("sigmas")
        if not isinstance(sigmas, list) or len(sigmas) != 3:
            raise ValueError("gaussian_noise.sigmas must be a list of exactly 3 floats")
        for sig in sigmas:
            if not isinstance(sig, (int, float)) or sig <= 0.0:
                raise ValueError(f"Gaussian noise sigma must be > 0.0, got {sig}")
        if not (sigmas[0] < sigmas[1] < sigmas[2]):
            raise ValueError(f"Gaussian noise sigmas must strictly increase: {sigmas}")
        if not isinstance(gnoise.get("seed"), int) or gnoise.get("seed") < 0:
            raise ValueError("gaussian_noise.seed must be a non-negative integer")

    # 5. Gaussian Blur (Optional)
    if "gaussian_blur" in config:
        gblur = config["gaussian_blur"]
        if not isinstance(gblur.get("enabled"), bool):
            raise ValueError("gaussian_blur.enabled must be a boolean")
        k_sizes = gblur.get("kernel_sizes")
        if not isinstance(k_sizes, list) or len(k_sizes) != 3:
            raise ValueError("gaussian_blur.kernel_sizes must be a list of 3 odd integers")
        for k in k_sizes:
            if not isinstance(k, int) or k < 3 or k % 2 == 0:
                raise ValueError(f"Gaussian blur kernel size must be odd integer >= 3, got {k}")
        if not (k_sizes[0] < k_sizes[1] < k_sizes[2]):
            raise ValueError(f"Gaussian blur kernel sizes must strictly increase: {k_sizes}")
        b_sigmas = gblur.get("sigmas")
        if not isinstance(b_sigmas, list) or len(b_sigmas) != 3:
            raise ValueError("gaussian_blur.sigmas must be a list of 3 floats")
        for b_sig in b_sigmas:
            if not isinstance(b_sig, (int, float)) or b_sig <= 0.0:
                raise ValueError(f"Gaussian blur sigma must be > 0.0, got {b_sig}")
        if not (b_sigmas[0] < b_sigmas[1] < b_sigmas[2]):
            raise ValueError(f"Gaussian blur sigmas must strictly increase: {b_sigmas}")

    # 6. Contrast Changes (Optional)
    if "contrast" in config:
        contrast = config["contrast"]
        if not isinstance(contrast.get("enabled"), bool):
            raise ValueError("contrast.enabled must be a boolean")
        factors = contrast.get("factors")
        if not isinstance(factors, list) or len(factors) != 3:
            raise ValueError("contrast.factors must be a list of 3 floats")
        for f in factors:
            if not isinstance(f, (int, float)) or f <= 0.0:
                raise ValueError(f"Contrast factor must be > 0.0, got {f}")

    # 7. Controlled Centered Cropping (Optional)
    if "crop" in config:
        crop = config["crop"]
        if not isinstance(crop.get("enabled"), bool):
            raise ValueError("crop.enabled must be a boolean")
        ratios = crop.get("ratios")
        if not isinstance(ratios, list) or len(ratios) != 3:
            raise ValueError("crop.ratios must be a list of 3 floats")
        for r in ratios:
            if not isinstance(r, (int, float)) or r <= 0.0 or r >= 1.0:
                raise ValueError(f"Crop ratio must be in (0.0, 1.0), got {r}")
        if not (ratios[0] > ratios[1] > ratios[2]):
            raise ValueError(f"Crop ratios must strictly decrease with severity: {ratios}")


def get_enabled_conditions(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Enumerate all experimental conditions enabled by the configuration.

    Always starts with clean reference (severity 0).
    For core configuration, produces 13 conditions per checkpoint:
      - 1 clean
      - 3 JPEG (qualities 80, 50, 20)
      - 3 Resize (scales 0.75, 0.50, 0.25)
      - 3 Brightness Dark (factors 0.80, 0.60, 0.40)
      - 3 Brightness Bright (factors 1.20, 1.40, 1.60)
    """
    if config is None:
        config = get_default_robustness_config()
    validate_robustness_config(config)

    conditions = [
        {
            "transformation": "clean",
            "severity": 0,
            "direction": "none",
            "parameter_name": "none",
            "parameter_value": None,
            "condition_id": "clean",
        }
    ]

    # 1. JPEG
    if config.get("jpeg", {}).get("enabled", False):
        qualities = config["jpeg"]["qualities"]
        for idx, q in enumerate(qualities, start=1):
            conditions.append({
                "transformation": "jpeg",
                "severity": idx,
                "direction": "none",
                "parameter_name": "quality",
                "parameter_value": q,
                "condition_id": f"jpeg_sev{idx}_q{q}",
            })

    # 2. Resize
    if config.get("resize", {}).get("enabled", False):
        scales = config["resize"]["scales"]
        for idx, s in enumerate(scales, start=1):
            conditions.append({
                "transformation": "resize",
                "severity": idx,
                "direction": "none",
                "parameter_name": "scale",
                "parameter_value": s,
                "condition_id": f"resize_sev{idx}_s{int(s*100)}",
            })

    # 3. Brightness (Darkening & Brightening evaluated separately)
    if config.get("brightness", {}).get("enabled", False):
        darker_factors = config["brightness"]["darker"]
        for idx, factor in enumerate(darker_factors, start=1):
            conditions.append({
                "transformation": "brightness_dark",
                "severity": idx,
                "direction": "darker",
                "parameter_name": "factor",
                "parameter_value": factor,
                "condition_id": f"brightness_dark_sev{idx}_f{int(factor*100)}",
            })

        brighter_factors = config["brightness"]["brighter"]
        for idx, factor in enumerate(brighter_factors, start=1):
            conditions.append({
                "transformation": "brightness_bright",
                "severity": idx,
                "direction": "brighter",
                "parameter_name": "factor",
                "parameter_value": factor,
                "condition_id": f"brightness_bright_sev{idx}_f{int(factor*100)}",
            })

    # 4. Gaussian Noise (Optional)
    if config.get("gaussian_noise", {}).get("enabled", False):
        sigmas = config["gaussian_noise"]["sigmas"]
        noise_seed = config["gaussian_noise"].get("seed", 42)
        for idx, sigma in enumerate(sigmas, start=1):
            conditions.append({
                "transformation": "gaussian_noise",
                "severity": idx,
                "direction": "none",
                "parameter_name": "sigma",
                "parameter_value": sigma,
                "seed": noise_seed,
                "condition_id": f"gaussian_noise_sev{idx}_sig{int(sigma)}",
            })

    # 5. Gaussian Blur (Optional)
    if config.get("gaussian_blur", {}).get("enabled", False):
        k_sizes = config["gaussian_blur"]["kernel_sizes"]
        b_sigmas = config["gaussian_blur"]["sigmas"]
        for idx, (k, s) in enumerate(zip(k_sizes, b_sigmas), start=1):
            conditions.append({
                "transformation": "gaussian_blur",
                "severity": idx,
                "direction": "none",
                "parameter_name": "kernel_sigma",
                "parameter_value": {"kernel_size": k, "sigma": s},
                "condition_id": f"gaussian_blur_sev{idx}_k{k}_sig{int(s)}",
            })

    # 6. Contrast (Optional)
    if config.get("contrast", {}).get("enabled", False):
        factors = config["contrast"]["factors"]
        for idx, f in enumerate(factors, start=1):
            conditions.append({
                "transformation": "contrast",
                "severity": idx,
                "direction": "none",
                "parameter_name": "factor",
                "parameter_value": f,
                "condition_id": f"contrast_sev{idx}_f{int(f*100)}",
            })

    # 7. Centered Crop (Optional)
    if config.get("crop", {}).get("enabled", False):
        ratios = config["crop"]["ratios"]
        for idx, r in enumerate(ratios, start=1):
            conditions.append({
                "transformation": "crop",
                "severity": idx,
                "direction": "none",
                "parameter_name": "ratio",
                "parameter_value": r,
                "condition_id": f"crop_sev{idx}_r{int(r*100)}",
            })

    return conditions


def save_robustness_config(
    config: Dict[str, Any],
    output_dir: Path,
    experiment_id: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save configuration snapshots (JSON and TXT) in experiment directory.

    Ensures experiment parameters are frozen and permanently recorded alongside outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_robustness_config(config)
    conditions = get_enabled_conditions(config)

    snapshot = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "robustness_config": config,
        "enabled_condition_count": len(conditions),
        "conditions": conditions,
    }
    if extra_metadata:
        snapshot["metadata"] = extra_metadata

    # 1. JSON snapshot
    json_path = output_dir / "config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # 2. Plain text overview
    txt_path = output_dir / "config.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Approach 2: Robustness Evaluation Configuration — {experiment_id}\n")
        f.write(f"Timestamp: {snapshot['timestamp']}\n")
        f.write(f"Total Conditions per Checkpoint: {len(conditions)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Transformation Settings ---\n")
        for name, params in config.items():
            f.write(f"[{name.upper()}]\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("--- Evaluated Conditions (per checkpoint) ---\n")
        for i, c in enumerate(conditions):
            f.write(
                f"  {i+1:02d}. id={c['condition_id']:<32} "
                f"transform={c['transformation']:<16} "
                f"sev={c['severity']} "
                f"dir={c['direction']:<8} "
                f"val={str(c['parameter_value'])}\n"
            )

    return json_path
