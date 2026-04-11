import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from app.config import Settings, get_settings

import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

IMG_SIZE = 256
COMMON_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_initial_dropout_rate(model_type: str) -> float:
    rates = {
        "efficientnet_b0": 0.40,
        "efficientnet_b1": 0.40,
        "efficientnet_b2": 0.45,
    }
    return rates.get(model_type.lower(), 0.30)

def get_model_by_type(model_type: str, num_classes: int = 2):
    model_type = model_type.lower()
    
    if model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    elif model_type == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    elif model_type == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Replace classifier head for binary classification with dropout
    in_features = model.classifier[1].in_features
    dropout = get_initial_dropout_rate(model_type)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    
    return model

MODEL_CACHE: Dict[str, "ModelBundle"] = {}

# Model type to canonical name mapping
MODEL_TYPE_NAMES = {
    "efficientnet_b0": "EfficientNetB0",
    "efficientnet_b1": "EfficientNetB1",
    "efficientnet_b2": "EfficientNetB2",
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

    # Check for model type in filename — check specific variants before generic prefix
    if "efficientnet_b2" in name:
        model = get_model_by_type('efficientnet_b2')
        model_name = "EfficientNetB2"
    elif "efficientnet_b1" in name:
        model = get_model_by_type('efficientnet_b1')
        model_name = "EfficientNetB1"
    elif "efficientnet" in name:
        model = get_model_by_type('efficientnet_b0')
        model_name = "EfficientNetB0"
    else:
        # Default fallback to EfficientNet-B0 (lightweight)
        model = get_model_by_type('efficientnet_b0')
        model_name = "EfficientNetB0"

    return model, model_name

def adapt_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], model_name: str) -> Dict[str, torch.Tensor]:
    """
    Adapt legacy state dicts to match current model architecture.
    Handles the transition from Linear classifier to Sequential(Dropout, Linear).
    """
    new_state_dict = state_dict.copy()
    
    # Check for EfficientNet (classifier.1.weight -> classifier.1.1.weight)
    if "EfficientNet" in model_name:
        if "classifier.1.weight" in state_dict and "classifier.1.1.weight" not in state_dict:
            if hasattr(model, "classifier") and isinstance(model.classifier[1], torch.nn.Sequential):
                print(f"[Model Service] Adapting legacy {model_name} checkpoint: classifier.1 -> classifier.1.1")
                new_state_dict["classifier.1.1.weight"] = state_dict["classifier.1.weight"]
                new_state_dict["classifier.1.1.bias"] = state_dict["classifier.1.bias"]
                del new_state_dict["classifier.1.weight"]
                del new_state_dict["classifier.1.bias"]

    return new_state_dict


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

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"[Model Service] Failed to load checkpoint {model_path}: {e}")
        return None

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

    # Adapt state dict if necessary (handles added dropout layers)
    state_dict = adapt_state_dict(model, state_dict, model_name)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Model Service] Strict loading failed for {model_name}: {e}")
        raise RuntimeError(
            f"Checkpoint mismatch for {model_name}. The saved weights do not match the "
            f"expected architecture. Ensure the model type matches the checkpoint."
        ) from e

    model.eval()
    model.to(device)

    bundle = ModelBundle(model=model, device=device, class_names=settings.class_names, model_name=model_name)
    MODEL_CACHE[path_str] = bundle
    return bundle
