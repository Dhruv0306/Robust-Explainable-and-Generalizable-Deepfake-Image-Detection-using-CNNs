"""
Utilities: reproducibility, device selection, logging
"""
import random
import sys
import numpy as np
import torch
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress torch's noisy CUDA warning; get_device() reports the usable fallback clearly.
warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.cuda\..*')


def _cuda_is_usable() -> bool:
    """Return True only when CUDA kernels can actually run."""
    try:
        if not torch.cuda.is_available():
            return False
        # Smoke-test catches GPUs newer than the installed PyTorch CUDA build.
        _ = torch.zeros(1, device="cuda") + 1
        return True
    except RuntimeError:
        return False


def ensure_gpu_torch_if_needed():
    """
    If a GPU is present but CPU-only torch is installed, reinstall CUDA torch.
    No-op on CPU machines or when torch already has CUDA support.
    """
    import subprocess

    has_gpu = torch.cuda.is_available()
    torch_cuda = torch.version.cuda

    logging.info(f"PyTorch version: {torch.__version__}, CUDA build: {torch_cuda or 'CPU-only'}")
    if not has_gpu and torch_cuda is None:
        logging.info("No CUDA GPU detected; keeping CPU-only PyTorch")
        return
    if torch_cuda is not None:
        logging.info("PyTorch already has CUDA support")
        return

    logging.warning("CUDA GPU detected but CPU-only PyTorch is installed; reinstalling CUDA PyTorch")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])
    logging.error("PyTorch was reinstalled. Restart this command so the new torch build is loaded.")
    sys.exit(0)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_is_usable():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cpu: bool = False) -> tuple[torch.device, dict]:
    """
    Select device and return device + info dict.
    Returns: (device, info_dict)
    """
    cuda_available = torch.cuda.is_available()
    cuda_usable = False if prefer_cpu else _cuda_is_usable()
    info = {
        "cuda_available": cuda_available,
        "cuda_usable": cuda_usable,
        "device_type": None,
        "device_name": None,
        "amp_enabled": False,
    }

    if prefer_cpu or not cuda_usable:
        device = torch.device("cpu")
        info["device_type"] = "cpu"
        info["device_name"] = "CPU"
        if cuda_available and not prefer_cpu:
            logging.warning("CUDA is visible but cannot run kernels; using CPU")
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


def discover_approach1_checkpoints(output_dir: Optional[Path] = None) -> list[dict]:
    """
    Discover all 9 trained Approach 1 checkpoints from output directory.

    Returns:
        List of dicts with keys: model, seed, run_name, run_dir, checkpoint_path,
        config_path, input_size
    """
    import json
    if output_dir is None:
        from config import OUTPUT_ROOT
        output_dir = OUTPUT_ROOT

    output_dir = Path(output_dir)
    checkpoints = []

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_file = run_dir / "config.json"
        ckpt_file = run_dir / "best_checkpoint.pth"

        if cfg_file.exists() and ckpt_file.exists():
            try:
                with open(cfg_file, "r") as f:
                    cfg = json.load(f)
                model_name = cfg.get("model")
                seed = cfg.get("seed")
                input_size = cfg.get("input_size")
                run_name = cfg.get("run_name", run_dir.name)

                if model_name and seed is not None:
                    checkpoints.append({
                        "model": model_name,
                        "seed": int(seed),
                        "run_name": run_name,
                        "run_dir": run_dir,
                        "checkpoint_path": ckpt_file,
                        "config_path": cfg_file,
                        "input_size": input_size,
                    })
            except Exception as e:
                logging.warning(f"Failed to inspect run directory {run_dir}: {e}")

    # Sort deterministically by model then seed
    checkpoints.sort(key=lambda x: (x["model"], x["seed"]))
    return checkpoints


def get_approach1_checkpoint(model: str, seed: int, output_dir: Optional[Path] = None) -> dict:
    """Retrieve specific Approach 1 checkpoint by model name and seed."""
    checkpoints = discover_approach1_checkpoints(output_dir)
    for ckpt in checkpoints:
        if ckpt["model"] == model and ckpt["seed"] == seed:
            return ckpt
    raise FileNotFoundError(f"Approach 1 checkpoint not found for model={model}, seed={seed}")
