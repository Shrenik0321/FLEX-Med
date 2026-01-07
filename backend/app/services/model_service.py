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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FED_LEARNING_PATH = PROJECT_ROOT / "federated_learning"
if FED_LEARNING_PATH.exists():
    sys.path.append(str(FED_LEARNING_PATH))

from flex_med.task import (  # type: ignore  # added to sys.path above
    COMMON_TRANSFORM,
    get_model_by_type,
)

MODEL_CACHE: Optional["ModelBundle"] = None


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
    name = model_path.name
    # Fallback to ResNet
    model = get_model_by_type('resnet18')
    model_name = "ResNet18"
    if "model_client_" in name:
        try:
            client_id = int(name.split("model_client_")[1].split(".")[0])
            mod = client_id % 3
            if mod == 0:
                model = get_model_by_type('resnet18')
                model_name = "ResNet18"
            elif mod == 1:
                model = get_model_by_type('mobilenet_v2')
                model_name = "MobileNetV2"
            else:
                model = get_model_by_type('efficientnet_b3')
                model_name = "EfficientNet-B3"
        except Exception:
            pass
    return model, model_name

def load_model(settings: Settings) -> ModelBundle:
    """Load and cache the model."""
    global MODEL_CACHE
    if MODEL_CACHE:
        return MODEL_CACHE

    device_str = settings.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model, model_name = select_model(settings.model_path)
    checkpoint = torch.load(settings.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    MODEL_CACHE = ModelBundle(model=model, device=device, class_names=settings.class_names, model_name=model_name)
    return MODEL_CACHE

def predict_image(image: Image.Image, settings: Optional[Settings] = None) -> dict:
    """Run prediction on a PIL image."""
    settings = settings or get_settings()
    bundle = load_model(settings)
    label, confidence, class_idx, all_probs = bundle.predict(image.convert("RGB"))
    return {
        "prediction": label,
        "label": label,  # Keep for backward compatibility
        "confidence": confidence,
        "class": class_idx,
        "model": bundle.model_name,
        "model_path": str(settings.model_path),
        "device": str(bundle.device),
        "all_probabilities": all_probs,
    }

def predict_random(settings: Optional[Settings] = None) -> dict:
    """Pick a random image from the configured folder and predict."""
    settings = settings or get_settings()
    if not settings.public_data_path:
        raise ValueError("public_data_path is not configured.")

    data_path = settings.public_data_path
    ds = datasets.ImageFolder(root=data_path, transform=COMMON_TRANSFORM)
    if len(ds) == 0:
        raise ValueError(f"No images found under {data_path}")

    idx = random.randint(0, len(ds) - 1)
    img_tensor, label_idx = ds[idx]
    img = Image.fromarray(np.uint8(img_tensor.mul(255).permute(1, 2, 0).numpy()))

    bundle = load_model(settings)
    with torch.no_grad():
        outputs = bundle.model(img_tensor.unsqueeze(0).to(bundle.device))
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        all_probs = {bundle.class_names[i]: float(probs[i].item()) for i in range(len(bundle.class_names))}

    return {
        "prediction": bundle.class_names[pred_idx],
        "label": bundle.class_names[pred_idx],  # Keep for backward compatibility
        "confidence": confidence,
        "class": pred_idx,
        "model": bundle.model_name,
        "model_path": str(settings.model_path),
        "device": str(bundle.device),
        "all_probabilities": all_probs,
        "true_label": bundle.class_names[label_idx] if label_idx < len(bundle.class_names) else label_idx,
        "sample_index": idx,
    }

def get_gradcam_target_layers(model: torch.nn.Module, model_name: str):
    """
    Resolve target layers for Grad-CAM based on architecture.
    """
    if "ResNet" in model_name:
        return [model.layer4[-1]]
    elif "MobileNet" in model_name:
        return [model.features[-1]]
    elif "EfficientNet" in model_name:
        return [model.features[-1]]
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")
