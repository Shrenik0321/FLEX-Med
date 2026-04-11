"""Unified inference endpoint with prediction."""
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
            
            # 3.5 Generate Grad-CAM explanation
            try:
                from app.services.xai_service import generate_gradcam
                gradcam_base64 = generate_gradcam(bundle.model, image, bundle.model_name, bundle.device)
                logger.info("Grad-CAM generated successfully.")
            except Exception as xai_e:
                logger.warning(f"Grad-CAM generation failed: {xai_e}")
                gradcam_base64 = None

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            ) from e

        # 4. Return unified response
        return JSONResponse(
            status_code=200,
            content={
                "prediction": label,
                "confidence": confidence,
                "class": class_idx,
                "all_probabilities": all_probs,
                "gradcam_image": gradcam_base64,
                "client_id": client_id,
                "model": bundle.model_name,
                "model_path": str(client_model_path),
                "device": str(bundle.device)
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected server error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
