import io
import base64
import numpy as np
from PIL import Image
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from flex_med.task import (  # type: ignore  # added to sys.path above
    COMMON_TRANSFORM,
)


# -----------------------------
# Unnormalize ImageNet tensors
# -----------------------------
def unnormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Input: (3, H, W) normalized tensor
    Output: (H, W, 3) numpy image in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = tensor.detach().cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


# -----------------------------
# Grad-CAM generator
# -----------------------------
def generate_gradcam_base64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layers: list,
    class_idx: int,
) -> str:
    """
    Generate Grad-CAM and return base64 PNG.
    input_tensor: (1, 3, H, W)
    """
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = unnormalize_tensor(input_tensor[0])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    pil_img = Image.fromarray(cam_image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_lime_base64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    device: torch.device,
    num_samples: int = 1000,
) -> str:
    """
    Generate LIME explanation and return base64 PNG.
    """

    # LIME expects numpy image in [0,255]
    img = unnormalize_tensor(input_tensor[0])
    img_uint8 = (img * 255).astype(np.uint8)

    def predict_fn(images):
        """
        images: List[np.ndarray] of shape (H, W, 3)
        returns: probabilities
        """
        tensors = torch.stack(
            [
                COMMON_TRANSFORM(Image.fromarray(im)).to(device)
                for im in images
            ]
        )

        with torch.no_grad():
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_uint8,
        classifier_fn=predict_fn,
        labels=(class_idx,),
        hide_color=0,
        num_samples=num_samples,
    )

    lime_img, _ = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    lime_vis = mark_boundaries(lime_img / 255.0, _)

    pil_img = Image.fromarray((lime_vis * 255).astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")