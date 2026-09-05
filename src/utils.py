"""
Utilities: reproducibility, device selection, logging
"""
import random
import numpy as np
import torch
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress CUDA capability warning for newer GPUs (e.g., RTX 5050 sm_120)
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA capability.*')


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cpu: bool = False) -> tuple[torch.device, dict]:
    """
    Select device and return device + info dict.
    Returns: (device, info_dict)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_type": None,
        "device_name": None,
        "amp_enabled": False,
    }

    if prefer_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        info["device_type"] = "cpu"
        info["device_name"] = "CPU"
    else:
        device = torch.device("cuda")
        info["device_type"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["amp_enabled"] = True  # AMP enabled when CUDA available

    return device, info


def setup_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """Setup logging to console and optionally to file"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def get_batch_size(device: torch.device, model_input_size: int, override: Optional[int] = None) -> int:
    """
    Determine batch size based on device and input size.
    Override takes precedence if provided.
    """
    if override is not None:
        return override

    from config import BATCH_SIZE_GPU, BATCH_SIZE_CPU

    if device.type == "cuda":
        # Conservative GPU batch size (can be tuned per GPU)
        if model_input_size >= 299:
            return max(16, BATCH_SIZE_GPU // 2)
        return BATCH_SIZE_GPU
    else:
        return BATCH_SIZE_CPU


def save_config(config_dict: dict, output_dir: Path):
    """Save config as both JSON and TXT"""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # TXT
    with open(output_dir / "config.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Approach 1: CNN Baseline Configuration\n")
        f.write("=" * 80 + "\n\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
