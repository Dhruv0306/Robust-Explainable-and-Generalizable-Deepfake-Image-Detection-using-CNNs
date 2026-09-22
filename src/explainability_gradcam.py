"""
Approach 3: Model Loading and Grad-CAM Generation Pipeline.
Handles checkpoint initialization, Grad-CAM extraction targeting the fake-class logit,
map normalization and upsampling, disk caching (.npy), and memory-safe batch-size-1 inference.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import create_model, get_target_layer, get_model_input_size
from dataset import get_transforms
from seed_selector import select_best_seed


class GradCAMGenerator:
    """
    Orchestrates memory-safe Grad-CAM generation for deepfake detection models.
    Guarantees:
    1. Gradients remain enabled during CAM generation (model.eval() without no_grad).
    2. Fixed target: always targets index 0 (fake-class single output logit).
    3. Output normalized to [0.0, 1.0] float32 arrays.
    4. Upsampled to original face crop (H, W) resolution.
    5. Caching as .npy files with metadata integrity.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path,
        device: torch.device,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.input_size = get_model_input_size(model_name)
        self.transform = get_transforms(model_name, is_train=False)

        self.model = create_model(model_name, pretrained=False).to(self.device)
        self._load_checkpoint()
        self.model.eval()

        self.target_layers = get_target_layer(model_name, self.model)
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.targets = [ClassifierOutputTarget(0)]
        self._closed = False

    def _load_checkpoint(self) -> None:
        """Load weights from checkpoint file with fallback key handling."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        logging.info(
            "Loading checkpoint for %s from %s",
            self.model_name,
            self.checkpoint_path,
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
        self.model.load_state_dict(state_dict)
        del checkpoint, state_dict

    def predict(self, face_crop_rgb: np.ndarray) -> Tuple[float, int]:
        """Run forward inference on a face crop in inference mode."""
        tensor = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                logit = self.model(tensor).squeeze()
                prob = torch.sigmoid(logit).item()
            return float(prob), int(prob >= 0.5)
        finally:
            # The returned values are Python scalars, so the input tensor and
            # forward intermediates no longer need to remain referenced here.
            del tensor
            if "logit" in locals():
                del logit

    def generate_cam(
        self,
        face_crop_rgb: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate a normalized Grad-CAM map targeting the fake-class logit."""
        if target_shape is None:
            target_shape = face_crop_rgb.shape[:2]

        tensor = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
        try:
            # Gradients must remain enabled for Grad-CAM.
            grayscale_cam = self.cam(
                input_tensor=tensor,
                targets=self.targets,
            )
            cam_map = grayscale_cam[0].astype(np.float32)

            cam_min, cam_max = float(cam_map.min()), float(cam_map.max())
            if cam_max - cam_min > 1e-7:
                cam_map = (cam_map - cam_min) / (cam_max - cam_min)
            else:
                cam_map = np.zeros_like(cam_map, dtype=np.float32)

            if cam_map.shape != target_shape:
                cam_map = cv2.resize(
                    cam_map,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                cam_map = np.clip(cam_map, 0.0, 1.0)

            return cam_map.astype(np.float32)
        finally:
            # pytorch-grad-cam handles its internal activation/gradient state,
            # while this releases our frame-local tensor reference immediately.
            del tensor
            if "grayscale_cam" in locals():
                del grayscale_cam

    def get_or_generate_cam(
        self,
        face_crop_rgb: np.ndarray,
        cache_path: Optional[Path] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Get Grad-CAM from cache or compute and save it."""
        if target_shape is None:
            target_shape = face_crop_rgb.shape[:2]

        if cache_path is not None and cache_path.exists():
            try:
                cached = np.load(cache_path)
                if cached.shape == target_shape and np.isfinite(cached).all():
                    result = cached.astype(np.float32)
                    del cached
                    return result
                del cached
            except Exception as exc:
                logging.warning(
                    "Cache read error for %s: %s. Recomputing.",
                    cache_path,
                    exc,
                )

        cam_map = self.generate_cam(face_crop_rgb, target_shape=target_shape)

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, cam_map)
            except Exception as exc:
                logging.warning(
                    "Failed to save Grad-CAM cache to %s: %s",
                    cache_path,
                    exc,
                )

        return cam_map

    def close(self) -> None:
        """Release Grad-CAM hooks and model references at model lifecycle end."""
        if self._closed:
            return

        try:
            cam = getattr(self, "cam", None)
            if cam is not None:
                activations_and_grads = getattr(cam, "activations_and_grads", None)
                if activations_and_grads is not None:
                    release = getattr(activations_and_grads, "release", None)
                    if callable(release):
                        release()
                self.cam = None
        except Exception as exc:
            logging.warning("Failed to release Grad-CAM hooks: %s", exc)
        finally:
            self.targets = None
            if getattr(self, "model", None) is not None:
                self.model.zero_grad(set_to_none=True)
                self.model = None
            self.target_layers = None
            self._closed = True

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_gradcam_generator(
    model_name: str,
    output_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    target_seed: Optional[int] = None,
) -> Tuple[GradCAMGenerator, Dict[str, Any]]:
    """Convenience factory to select best seed and instantiate GradCAMGenerator."""
    run_info = select_best_seed(
        model_name=model_name,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        target_seed=target_seed,
    )
    generator = GradCAMGenerator(
        model_name=model_name,
        checkpoint_path=Path(run_info["checkpoint_path"]),
        device=device,
    )
    return generator, run_info


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import OUTPUT_ROOT, CHECKPOINT_ROOT
    from utils import get_device

    logging.basicConfig(level=logging.INFO)
    dev, dev_info = get_device()
    print(f"Testing GradCAMGenerator on device: {dev} ({dev_info['device_name']})")

    for m in ["xception", "efficientnet_b0", "resnet50"]:
        gen, info = load_gradcam_generator(m, OUTPUT_ROOT, CHECKPOINT_ROOT, dev)
        dummy_crop = np.random.randint(0, 256, (150, 120, 3), dtype=np.uint8)
        prob, pred = gen.predict(dummy_crop)
        cam = gen.generate_cam(dummy_crop)
        print(
            f"[{m.upper()} seed {info['seed']}] prob={prob:.4f}, pred={pred}, "
            f"cam shape={cam.shape}, min={cam.min():.4f}, max={cam.max():.4f}"
        )
        assert cam.shape == (150, 120), f"Expected (150, 120), got {cam.shape}"
        assert 0.0 <= cam.min() and cam.max() <= 1.0, (
            f"CAM out of range [0, 1]: min={cam.min()}, max={cam.max()}"
        )
        gen.close()
