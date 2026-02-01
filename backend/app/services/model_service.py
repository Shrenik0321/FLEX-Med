import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from app.config import Settings, get_settings

# Make the local 0 package importable (lives in ../federated_learning)
# backend/app/services/model_service.py -> services -> app -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
FED_LEARNING_PATH = BACKEND_ROOT / "federated_learning"
if FED_LEARNING_PATH.exists():
    sys.path.append(str(FED_LEARNING_PATH))

from typing import Dict, Optional, Tuple

from flex_med.task import (  # type: ignore  # added to sys.path above
    COMMON_TRANSFORM,
    get_model_by_type,
)

MODEL_CACHE: Dict[str, "ModelBundle"] = {}

# Model type to canonical name mapping
MODEL_TYPE_NAMES = {
    "resnet50": "ResNet50",
    "mobilenet_v2": "MobileNetV2",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
}


class ModelBundle:
    """Container for the model and related metadata."""
    def __init__(self, model: torch.nn.Module, device: torch.device, class_names, model_name: str):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model_name = model_name

    def predict(self, img: Image.Image) -> Tuple[str, float, int, dict]:
        """Run inference on a PIL image and return (label, confidence, class_index, all_probabilities)."""
        tensor = COMMON_TRANSFORM(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())
            all_probs = {self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))}
        return self.class_names[idx], confidence, idx, all_probs


def select_model(model_path: Path) -> Tuple[torch.nn.Module, str]:
    """Pick architecture based on filename convention. Returns (model, model_name)."""
    name = model_path.name.lower()

    # Check for model type in filename
    if "mobilenet" in name:
        model = get_model_by_type('mobilenet_v2')
        model_name = "MobileNetV2"
    elif "efficientnet" in name or "effnet" in name:
        model = get_model_by_type('efficientnet_b0')
        model_name = "EfficientNet-B0"
    elif "resnet" in name:
        model = get_model_by_type('resnet50')
        model_name = "ResNet50"
    elif "densenet" in name:
        model = get_model_by_type('densenet121')
        model_name = "DenseNet121"
    else:
        # Default fallback to MobileNetV2 (lightweight)
        model = get_model_by_type('mobilenet_v2')
        model_name = "MobileNetV2"

    return model, model_name

def load_model(settings: Settings, model_path: Optional[Path] = None) -> ModelBundle:
    """Load and cache the model by its path."""
    global MODEL_CACHE
    
    # Use provided model_path or fallback to settings
    target_path = model_path
    path_str = str(target_path)

    if path_str in MODEL_CACHE:
        return MODEL_CACHE[path_str]

    device_str = settings.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint = torch.load(model_path, map_location=device)

    # Handle checkpoint format (with metadata) vs legacy format (direct state_dict)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # New checkpoint format with metadata
        state_dict = checkpoint["state_dict"]
        model_type = checkpoint.get("model_type", None)

        # Use model_type from checkpoint if available, otherwise detect from filename
        if model_type:
            model = get_model_by_type(model_type)
            # Use canonical name from mapping, fallback to title case if not in mapping
            model_name = MODEL_TYPE_NAMES.get(model_type, model_type.replace("_", " ").title())
        else:
            model, model_name = select_model(model_path)
    else:
        # Legacy format: checkpoint IS the state_dict
        state_dict = checkpoint
        model, model_name = select_model(model_path)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    bundle = ModelBundle(model=model, device=device, class_names=settings.class_names, model_name=model_name)
    MODEL_CACHE[path_str] = bundle
    return bundle

def _detect_architecture_type(model: torch.nn.Module) -> str:
    """
    Detect architecture type by inspecting model structure.

    Returns:
        Architecture type string (resnet, mobilenet, efficientnet, densenet, vgg, unknown)
    """
    model_class = model.__class__.__name__.lower()

    # Check by class name first
    if "resnet" in model_class:
        return "resnet"
    elif "mobilenet" in model_class:
        return "mobilenet"
    elif "efficientnet" in model_class:
        return "efficientnet"
    elif "densenet" in model_class:
        return "densenet"
    elif "vgg" in model_class:
        return "vgg"

    # Check by structure (attributes)
    if hasattr(model, "layer4"):
        return "resnet"
    elif hasattr(model, "features") and hasattr(model, "classifier"):
        # Could be MobileNet, EfficientNet, DenseNet, or VGG
        # Check for specific DenseNet patterns
        if any("denseblock" in str(m).lower() for m in model.features.children()):
            return "densenet"
        # MobileNet typically has InvertedResidual blocks
        elif any("invertedresidual" in str(m.__class__).lower() for m in model.features.children()):
            return "mobilenet"
        # VGG has only Conv2d and pooling
        elif all(isinstance(m, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.ReLU, torch.nn.BatchNorm2d))
                 for m in model.features.children()):
            return "vgg"
        # Default to efficientnet for remaining feature-based architectures
        else:
            return "efficientnet"

    return "unknown"


def _find_last_conv_layer(model: torch.nn.Module):
    """
    Fallback method to find the last convolutional layer in the model.

    Returns:
        Last Conv2d layer found, or None if not found
    """
    last_conv = None

    def find_conv(module):
        nonlocal last_conv
        for child in module.children():
            if isinstance(child, torch.nn.Conv2d):
                last_conv = child
            else:
                find_conv(child)

    find_conv(model)
    return last_conv


def get_gradcam_target_layers(model: torch.nn.Module, model_name: str):
    """
    Resolve target layers for Grad-CAM based on architecture.

    Uses both model name and structure inspection for robust layer detection.

    Args:
        model: PyTorch model
        model_name: Model name string (e.g., "ResNet18", "MobileNetV2")

    Returns:
        List of target layers for Grad-CAM

    Raises:
        ValueError: If no suitable layer can be found
    """
    # Try name-based detection first (fast path)
    if "ResNet" in model_name and hasattr(model, "layer4"):
        return [model.layer4[-1]]
    elif "MobileNet" in model_name and hasattr(model, "features"):
        return [model.features[-1]]
    elif "EfficientNet" in model_name and hasattr(model, "features"):
        return [model.features[-1]]
    elif "DenseNet" in model_name and hasattr(model, "features"):
        return [model.features[-1]]
    elif "VGG" in model_name and hasattr(model, "features"):
        return [model.features[-1]]

    # Fallback: detect architecture by structure
    arch_type = _detect_architecture_type(model)

    if arch_type == "resnet":
        if hasattr(model, "layer4"):
            return [model.layer4[-1]]
    elif arch_type in ["mobilenet", "efficientnet", "densenet", "vgg"]:
        if hasattr(model, "features"):
            return [model.features[-1]]

    # Last resort: find last Conv2d layer
    last_conv = _find_last_conv_layer(model)
    if last_conv is not None:
        import logging
        logging.warning(
            f"Could not detect architecture for '{model_name}'. "
            f"Using last Conv2d layer as fallback."
        )
        return [last_conv]

    # No suitable layer found
    raise ValueError(
        f"Could not find suitable Grad-CAM target layer for model '{model_name}'. "
        f"Detected architecture: {arch_type}. "
        f"Model must have either 'layer4' (ResNet) or 'features' (MobileNet/EfficientNet/DenseNet/VGG) "
        f"or at least one Conv2d layer."
    )
