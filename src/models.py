"""
CNN model definitions: Xception, EfficientNet-B0, ResNet50 via timm.
"""
import torch
import torch.nn as nn
import timm
import logging


def create_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Create binary deepfake classification model.

    Args:
        model_name: 'xception', 'efficientnet_b0', or 'resnet50'
        pretrained: Use ImageNet pretrained weights

    Returns:
        Model with single logit output
    """
    from config import MODELS

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

    model_cfg = MODELS[model_name]
    timm_name = model_cfg["timm_name"]

    logging.info(f"Creating {model_name} (timm: {timm_name}, pretrained: {pretrained})")

    # Load pretrained model
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=1  # Binary classification, single logit
    )

    logging.info(f"Model created: {model_name}")
    return model


def get_model_input_size(model_name: str) -> int:
    """Get required input size for model"""
    from config import MODELS
    return MODELS[model_name]["input_size"]


def get_model_normalization(model_name: str) -> dict:
    """
    Get ImageNet normalization parameters for model.
    All timm models use ImageNet stats by default.
    """
    # ImageNet normalization (timm default)
    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)

    for model_name in ["xception", "efficientnet_b0", "resnet50"]:
        model = create_model(model_name, pretrained=False)
        input_size = get_model_input_size(model_name)

        # Test forward pass
        x = torch.randn(2, 3, input_size, input_size)
        out = model(x)
        print(f"{model_name}: input {x.shape} -> output {out.shape}")
        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
