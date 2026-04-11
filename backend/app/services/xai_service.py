import base64
import io
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from app.services.model_service import COMMON_TRANSFORM

import torch.nn as nn

def get_target_layer(model, model_name: str):
    """
    Get the target layer for Grad-CAM based on the model architecture.
    Validates that the target layer is suitable for spatial feature extraction.
    Only supports EfficientNet variants (b0, b1, b2) for this research.
    """
    target_layer = None
    
    if "EfficientNet" in model_name:
        # EfficientNet from torchvision: model.features[-1] is the last Conv2dNormActivation
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Only EfficientNet variants (b0, b1, b2) are supported for Grad-CAM in this research. Unsupported model: {model_name}")
    
    # Basic validation: Ensure the target layer has convolutional components
    has_conv = any(isinstance(m, nn.Conv2d) for m in target_layer.modules()) if isinstance(target_layer, nn.Module) else False
    if not has_conv and not isinstance(target_layer, nn.Conv2d):
        print(f"[Warning] Target layer for {model_name} may not contain spatial convolutions.")
        
    return [target_layer]

def generate_gradcam(model: torch.nn.Module, image: Image.Image, model_name: str, device: torch.device) -> str:
    """
    Generate a Grad-CAM heatmap for the given image and model, and return it as a base64 encoded string.
    """
    # Prepare the input tensor
    input_tensor = COMMON_TRANSFORM(image).unsqueeze(0).to(device)
    
    # Get the target layer
    target_layers = get_target_layer(model, model_name)
    
    # Initialize GradCAM
    # We use use_cuda if device is cuda
    use_cuda = device.type == 'cuda'
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Generate the CAM for the highest scoring class
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
    # Convert PIL image to numpy array in [0, 1] range for show_cam_on_image
    # First resize the original image to match the model's expected input size (256x256 based on COMMON_TRANSFORM)
    # Wait, COMMON_TRANSFORM resizes to IMG_SIZE. Let's get the size from the tensor.
    _, _, h, w = input_tensor.shape
    resized_img = image.resize((w, h))
    rgb_img = np.float32(resized_img) / 255.0
    
    # Overlay the heatmap
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Convert back to PIL Image
    cam_pil = Image.fromarray(np.uint8(255 * cam_image))
    
    # Encode to base64
    buffered = io.BytesIO()
    cam_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str
