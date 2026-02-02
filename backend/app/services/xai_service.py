import io
import base64
import random
import numpy as np
from PIL import Image
import torch

from pytorch_grad_cam import GradCAM, EigenCAM
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
# Model Quality Validation
# -----------------------------
def validate_model_quality(
    model: torch.nn.Module,
    confidence: float,
    min_confidence: float = 0.75,
) -> dict:
    """
    Validate if model quality is sufficient for reliable xAI.

    Args:
        model: The PyTorch model
        confidence: Prediction confidence score [0, 1]
        min_confidence: Minimum confidence threshold for reliable xAI

    Returns:
        dict with 'passed', 'confidence', 'warnings', 'recommendation'
    """
    warnings = []
    issues = []

    # Check confidence level
    if confidence < min_confidence:
        issues.append(
            f"Low prediction confidence ({confidence:.2%} < {min_confidence:.0%}). "
            "xAI visualizations may be unreliable."
        )
        warnings.append("Consider training model further or using transfer learning.")

    # Check if prediction is near random (50/50)
    if 0.45 <= confidence <= 0.55:
        issues.append(
            f"Prediction confidence near random ({confidence:.2%}). "
            "Model has not learned discriminative features."
        )
        warnings.append("Model may need more training data or better architecture.")

    passed = len(issues) == 0

    return {
        "passed": passed,
        "confidence": confidence,
        "warnings": warnings,
        "issues": issues,
        "recommendation": (
            "xAI should be reliable" if passed
            else "xAI may show incorrect/noisy attributions. Improve model first."
        )
    }


# -----------------------------
# Grad-CAM generator
# -----------------------------
def generate_gradcam_base64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layers: list,
    class_idx: int,
    eigen_smooth: bool = False,
    aug_smooth: bool = True,
) -> str:
    """
    Generate Grad-CAM and return base64 PNG.

    Args:
        model: PyTorch model
        input_tensor: (1, 3, H, W) preprocessed tensor
        target_layers: List of target layers for CAM
        class_idx: Target class index
        eigen_smooth: Use EigenCAM for smoother, less noisy heatmaps (default: False)
        aug_smooth: Average over augmentations for stability (default: True)

    Returns:
        Base64 encoded PNG image with Grad-CAM heatmap overlay
    """
    # Ensure model is in eval mode
    model.eval()

    # Ensure gradients are enabled for input
    input_tensor.requires_grad_(True)

    # Choose CAM method
    # GradCAM is generally better for class-discriminative localization
    # EigenCAM is better for object localization regardless of class
    CAMClass = EigenCAM if eigen_smooth else GradCAM
    cam = CAMClass(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
        aug_smooth=aug_smooth,  # Average over augmentations
        eigen_smooth=eigen_smooth,  # Additional smoothing for EigenCAM
    )[0]

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
    random_seed: int = 42,
    num_features: int = 5,
) -> str:
    """
    Generate LIME explanation and return base64 PNG.

    Args:
        model: PyTorch model
        input_tensor: Preprocessed tensor (1, 3, H, W)
        class_idx: Target class index
        device: torch device
        num_samples: Number of perturbation samples for LIME (default: 1000)
        random_seed: Random seed for reproducibility (default: 42)
        num_features: Number of super-pixel regions to highlight (default: 5)

    Returns:
        Base64 encoded PNG image with LIME visualization
    """
    # Set all random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # CRITICAL FIX: Keep in [0,1] space for consistency with model preprocessing
    # Unnormalize only for visualization, NOT for LIME perturbations
    img_rgb = unnormalize_tensor(input_tensor[0])  # [0, 1] range

    def predict_fn(images):
        """
        Classifier function for LIME.

        CRITICAL: images are in [0,1] range from LIME.
        Convert to PIL -> apply COMMON_TRANSFORM (same as training).

        Args:
            images: List[np.ndarray] of shape (H, W, 3) in [0, 1] range

        Returns:
            probabilities: (N, num_classes) numpy array
        """
        # Convert [0,1] numpy arrays to PIL Images (0-255 uint8)
        pil_images = [
            Image.fromarray((img * 255).astype(np.uint8))
            for img in images
        ]

        # Apply exact same transform as training
        tensors = torch.stack([
            COMMON_TRANSFORM(pil_img).to(device)
            for pil_img in pil_images
        ])

        with torch.no_grad():
            model.eval()
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    # Create LIME explainer with fixed random seed
    explainer = lime_image.LimeImageExplainer(random_state=random_seed)

    # Generate explanation on [0,1] image
    explanation = explainer.explain_instance(
        img_rgb,  # [0, 1] range for LIME
        classifier_fn=predict_fn,
        labels=(class_idx,),
        hide_color=0,
        num_samples=num_samples,
        random_seed=random_seed,  # Pass seed to explain_instance
    )

    # Get segmented image and mask
    lime_img, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )

    # Visualize with boundaries (lime_img already in [0,1] range)
    lime_vis = mark_boundaries(lime_img, mask)

    # Convert to PIL and encode
    pil_img = Image.fromarray((lime_vis * 255).astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")