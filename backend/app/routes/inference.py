"""Unified inference endpoint with prediction and explainable AI (xAI)."""
import io
import logging
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from supabase import Client as SupabaseClient

from app.config import get_settings, get_supabase_client, Settings
from app.services.model_service import (
    load_model,
    COMMON_TRANSFORM,
    get_gradcam_target_layers,
)
from app.services.xai_service import (
    generate_gradcam_base64,
    generate_lime_base64,
    validate_model_quality,
)

router = APIRouter(prefix="/inference", tags=["inference"])
logger = logging.getLogger(__name__)


@router.post("")
async def inference(
    client_id: int = Form(...),
    file: UploadFile = File(...),
    supabase: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings)
):
    """
    Unified inference endpoint: Prediction + Grad-CAM + LIME explanations.

    Processes an uploaded blood cell microscopy image and returns:
    - Prediction (Healthy vs ALL/Leukemia)
    - Confidence score
    - All class probabilities
    - Grad-CAM visualization (base64 encoded PNG)
    - LIME visualization (base64 encoded PNG)

    Args:
        file: Uploaded blood cell microscopy image (JPEG/PNG)

    Returns:
        JSON response with prediction and xAI visualizations

    Example Response:
        {
            "prediction": "ALL (Leukemia)",
            "confidence": 0.87,
            "class": 1,
            "all_probabilities": {
                "Healthy": 0.13,
                "ALL (Leukemia)": 0.87
            },
            "model": "MobileNetV2",
            "model_path": "/path/to/model.pt",
            "device": "cpu",
            "xai": {
                "gradcam": {
                    "image_base64": "iVBORw0KGgoAAAANS..."
                },
                "lime": {
                    "image_base64": "iVBORw0KGgoAAAANS..."
                }
            }
        }
    """
    try:
        logger.info(f"Received inference request for client ID {client_id}, file: {file.filename}")

        # 1. Fetch client and model path from database
        client_response = supabase.from_("clients").select("model_path").eq("id", client_id).execute()
        if not client_response.data:
            logger.error(f"Inference failed: Client {client_id} not found.")
            raise HTTPException(
                status_code=404,
                detail=f"Client with ID {client_id} not found."
            )
        
        client_model_path = client_response.data[0].get("model_path")
        if not client_model_path:
            logger.error(f"Inference failed: No model_path configured for client {client_id}.")
            raise HTTPException(
                status_code=404,
                detail=f"Client with ID {client_id} does not have a model path configured."
            )

        # 2. Load model
        from pathlib import Path
        bundle = load_model(settings, model_path=Path(client_model_path))

        if bundle is None or bundle.model is None:
            logger.critical("Inference failed: Model not loaded.")
            raise HTTPException(
                status_code=503,
                detail="Model not loaded on server. Check model path configuration."
            )

        # 2. Read and process image
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info(f"Image processed. Size: {image.size}")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            ) from e

        # 3. Run prediction
        try:
            label, confidence, class_idx, all_probs = bundle.predict(image)
            logger.info(f"Prediction: {label} ({confidence:.4f})")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            ) from e

        # 3.5. Validate model quality before generating xAI
        quality_check = validate_model_quality(
            model=bundle.model,
            confidence=confidence,
            min_confidence=settings.xai_min_confidence,
        )

        # Log warnings if model quality is questionable
        if quality_check["warnings"]:
            for warning in quality_check["warnings"]:
                logger.warning(f"xAI Quality Warning: {warning}")
        if quality_check["issues"]:
            for issue in quality_check["issues"]:
                logger.warning(f"xAI Quality Issue: {issue}")
            logger.warning(f"Recommendation: {quality_check['recommendation']}")

        # 4. Prepare input tensor for xAI
        try:
            input_tensor = COMMON_TRANSFORM(image).unsqueeze(0).to(bundle.device)
        except Exception as e:
            logger.error(f"Tensor transformation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to prepare image tensor: {str(e)}"
            ) from e

        # 5. Generate Grad-CAM
        try:
            target_layers = get_gradcam_target_layers(bundle.model, bundle.model_name)
            gradcam_base64 = generate_gradcam_base64(
                model=bundle.model,
                input_tensor=input_tensor,
                target_layers=target_layers,
                class_idx=class_idx,
                eigen_smooth=settings.xai_gradcam_eigen_smooth,
                aug_smooth=True,
            )
            logger.info("Grad-CAM generated successfully")
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Grad-CAM generation failed: {str(e)}"
            ) from e

        # 6. Generate LIME
        try:
            lime_base64 = generate_lime_base64(
                model=bundle.model,
                input_tensor=input_tensor,
                class_idx=class_idx,
                device=bundle.device,
                num_samples=settings.xai_lime_num_samples,
                random_seed=settings.xai_lime_random_seed,
                num_features=settings.xai_lime_num_features,
            )
            logger.info("LIME generated successfully")
        except Exception as e:
            logger.error(f"LIME generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LIME generation failed: {str(e)}"
            ) from e

        # 7. Return unified response
        return JSONResponse(
            status_code=200,
            content={
                "prediction": label,
                "confidence": confidence,
                "class": class_idx,
                "all_probabilities": all_probs,
                "client_id": client_id,
                "model": bundle.model_name,
                "model_path": str(client_model_path),
                "device": str(bundle.device),
                "xai": {
                    "gradcam": {
                        "image_base64": gradcam_base64
                    },
                    "lime": {
                        "image_base64": lime_base64
                    }
                }
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected server error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
