"""
PyTorch Dataset and DataLoader with augmentations.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import cv2
from pathlib import Path
import logging
from typing import Optional
import sys
sys.path.append(str(Path(__file__).parent))
from config import *


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection from manifest"""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            manifest_path: Path to manifest.csv
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
        """
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

        logging.info(f"{split} dataset: {len(self.df)} frames")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image (already face-cropped)
        img = cv2.imread(row["frame_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Label: 0=real, 1=fake
        label = 1.0 if row["label"] == "fake" else 0.0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32), row["video_id"]


def get_transforms(model_name: str, is_train: bool = False) -> transforms.Compose:
    """
    Get transforms for model.

    Args:
        model_name: Model name for input size
        is_train: Apply training augmentation
    """
    from models import get_model_input_size, get_model_normalization

    input_size = get_model_input_size(model_name)
    norm = get_model_normalization(model_name)

    transform_list = []

    # Resize
    transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.Resize((input_size, input_size)))

    # Training augmentation
    if is_train:
        transform_list.append(transforms.RandomHorizontalFlip(p=HFLIP_PROB))
        # Gaussian blur
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=GAUSSIAN_BLUR_KERNEL_SIZES[0], sigma=GAUSSIAN_BLUR_SIGMA_RANGE)],
                p=GAUSSIAN_BLUR_PROB
            )
        )

    # To tensor + normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=norm["mean"], std=norm["std"]))

    return transforms.Compose(transform_list)


def get_dataloader(
    manifest_path: Path,
    split: str,
    model_name: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Create DataLoader for split"""
    if shuffle is None:
        shuffle = (split == "train")

    transform = get_transforms(model_name, is_train=(split == "train"))
    dataset = DeepfakeDataset(manifest_path, split, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_class_weights(manifest_path: Path) -> torch.Tensor:
    """Compute class weights for training set"""
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]

    n_real = len(train_df[train_df["label"] == "real"])
    n_fake = len(train_df[train_df["label"] == "fake"])
    total = n_real + n_fake

    # Weight inversely proportional to class frequency
    weight_real = total / (2 * n_real)
    weight_fake = total / (2 * n_fake)

    logging.info(f"Class weights: real={weight_real:.3f}, fake={weight_fake:.3f}")

    return torch.tensor([weight_real, weight_fake], dtype=torch.float32)


class RobustnessDataset(Dataset):
    """
    Dataset for robustness evaluation under controlled corruptions.
    Loads raw face crop, dynamically applies corruption condition,
    then applies model-specific resize and ImageNet normalization.
    Preserves full manifest metadata (video_id, frame_path, category, original_frame_number).
    """

    def __init__(
        self,
        manifest_path: Path,
        model_name: str,
        split: str = "test",
        condition: Optional[dict] = None,
        deterministic_noise_seed: int = 42,
    ):
        from models import get_model_input_size, get_model_normalization

        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.model_name = model_name
        self.condition = condition or {
            "transformation": "clean",
            "severity": 0,
            "direction": "none",
            "parameter_name": "none",
            "parameter_value": None,
            "condition_id": "clean",
        }
        self.deterministic_noise_seed = deterministic_noise_seed

        input_size = get_model_input_size(model_name)
        norm = get_model_normalization(model_name)

        # Model-specific preprocessing: ToPIL -> Resize -> Tensor -> Normalize
        self.post_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ])
        logging.info(
            f"RobustnessDataset ({model_name}, {self.condition.get('condition_id', 'clean')}): {len(self.df)} frames"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_path = row["frame_path"]

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {frame_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply dynamic robustness transformation if not clean
        if self.condition and self.condition.get("severity", 0) > 0:
            from robustness import apply_condition
            import hashlib

            # Frame-level deterministic seed across architectures
            seed_str = f"{frame_path}_{self.condition['severity']}_{self.deterministic_noise_seed}"
            frame_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            img_rgb, _ = apply_condition(img_rgb, self.condition, seed=frame_seed)

        # Model-specific resize and normalization
        tensor_img = self.post_transform(img_rgb)

        label = 1.0 if row["label"] == "fake" else 0.0

        meta = {
            "video_id": str(row["video_id"]),
            "frame_path": str(frame_path),
            "category": str(row["category"]),
            "label": int(label),
            "original_frame_number": int(row["original_frame_number"])
            if "original_frame_number" in row and not pd.isna(row["original_frame_number"])
            else -1,
            "split": str(row.get("split", "test")),
        }

        return tensor_img, torch.tensor(label, dtype=torch.float32), meta


def get_robustness_dataloader(
    manifest_path: Path,
    model_name: str,
    split: str = "test",
    condition: Optional[dict] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    deterministic_noise_seed: int = 42,
) -> DataLoader:
    """Create DataLoader for robustness evaluation under controlled corruptions."""
    dataset = RobustnessDataset(
        manifest_path=manifest_path,
        model_name=model_name,
        split=split,
        condition=condition,
        deterministic_noise_seed=deterministic_noise_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle evaluation data
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    from utils import setup_logging

    setup_logging()

    # Test dataset loading
    manifest = MANIFESTS_ROOT / "manifest.csv"
    if manifest.exists():
        dataset = DeepfakeDataset(manifest, "train", get_transforms("xception", is_train=True))
        print(f"Dataset size: {len(dataset)}")

        img, label, video_id = dataset[0]
        print(f"Sample: img shape={img.shape}, label={label}, video_id={video_id}")
