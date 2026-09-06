"""
Training loop: AdamW + ReduceLROnPlateau + early stopping + AMP + checkpointing.
"""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent))
from config import *
from models import create_model, get_model_input_size
from dataset import get_dataloader, compute_class_weights
from utils import set_seed, get_device, get_batch_size, save_config


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    """Single training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward with AMP
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images).squeeze(1)  # (B,) logits
            loss = criterion(outputs, labels)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp):
    """Validation pass"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(dataloader, desc="Val"):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(
    model_name: str,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
    device: Optional[torch.device] = None,
    batch_size_override: Optional[int] = None,
) -> Dict:
    """
    Full training pipeline.

    Returns dict with training history and best metrics.
    """
    set_seed(seed)
    logging.info(f"Training {model_name} with seed {seed}")

    # Device
    if device is None:
        device, device_info = get_device()
    else:
        _, device_info = get_device()

    use_amp = device_info["amp_enabled"]
    logging.info(f"Device: {device_info['device_name']}, AMP: {use_amp}")

    # Model
    model = create_model(model_name, pretrained=True)
    model = model.to(device)

    # Batch size
    input_size = model.input_size if hasattr(model, 'input_size') else get_model_input_size(model_name)
    batch_size = get_batch_size(device, input_size, batch_size_override)
    logging.info(f"Batch size: {batch_size}")

    # Data
    train_loader = get_dataloader(manifest_path, "train", model_name, batch_size, NUM_WORKERS)
    val_loader = get_dataloader(manifest_path, "val", model_name, batch_size, NUM_WORKERS)

    # Loss with class weights
    class_weights = compute_class_weights(manifest_path).to(device)
    # BCEWithLogitsLoss expects pos_weight (weight for positive class = fake = class 1)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logging.info(f"pos_weight: {pos_weight:.3f}")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE)

    # AMP scaler
    scaler = GradScaler(device='cuda', enabled=use_amp)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        logging.info(f"Epoch {epoch}/{MAX_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logging.info(f"LR: {current_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            checkpoint_path = output_dir / "best_checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            logging.info(f"Saved best checkpoint at epoch {epoch}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    logging.info(f"Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate training curves
    try:
        from visualize import plot_training_curves
        plot_training_curves(history, output_dir / "training_curves.png")
        logging.info("Training curves saved")
    except Exception as e:
        logging.warning(f"Failed to generate training curves: {e}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


if __name__ == "__main__":
    from utils import setup_logging

    setup_logging()

    # Test training
    manifest = MANIFESTS_ROOT / "manifest.csv"
    output_dir = OUTPUT_ROOT / "test_run"

    if manifest.exists():
        train_model("xception", manifest, output_dir, seed=42)
