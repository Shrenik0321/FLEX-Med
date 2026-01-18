import io
import logging
import torch
import requests
from fastapi import APIRouter, File, HTTPException, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import os
from supabase import Client as SupabaseClient

from app.schemas.prediction import PredictionResponse
from app.config import get_settings, get_supabase_client
from app.services.model_service import (
    predict_image,
    load_model,
    COMMON_TRANSFORM,
    get_gradcam_target_layers,
)
from app.services.xai_service import generate_gradcam_base64
from app.services.xai_service import generate_lime_base64

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

# Load inference orchestrator URL from settings
# This URL should point to the ngrok tunnel from Colab (e.g., https://xxxx.ngrok-free.app)
# Update the INFERENCE_ORCHESTRATOR_URL environment variable in .env when starting a new Colab session

# Ngrok URL for local training orchestrator (update when Colab session changes)
INFERENCE_ORCHESTRATOR_URL = os.getenv(
    "LOCAL_TRAIN_ORCHESTRATOR_URL",
    "https://intraspinal-agape-deidra.ngrok-free.dev"
)

# ============================================================
# STANDARD PREDICTION (UNCHANGED)
# ============================================================
@router.post("/upload", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """Predict from an uploaded image. Matches the /inference endpoint format."""
    try:
        logger.info(f"Received inference request for: {file.filename}")

        # 1. Check Model Availability
        settings = get_settings()
        bundle = load_model(settings)
        if bundle is None or bundle.model is None:
            logger.critical("Inference failed: Model variable is None.")
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        # 2. Read & Process Image
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info(f"Image processed. Size: {image.size}")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file.") from e

        # 3. Run Prediction
        try:
            result = predict_image(image, settings)
            logger.info(f"Result: {result['prediction']} ({result['confidence']:.4f})")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed.") from e

        # 4. Return response
        return JSONResponse(
            status_code=200,
            content={
                "prediction": result["prediction"],
                "label": result.get("label", result["prediction"]),
                "confidence": result["confidence"],
                "class": result["class"],
                "model": result["model"],
                "model_path": result["model_path"],
                "device": result["device"],
                "all_probabilities": result["all_probabilities"],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected server error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# PREDICTION + XAI (PROXIED TO COLAB ORCHESTRATOR)
# ============================================================
@router.post("/upload-xai")
async def upload_xai_proxy(
    file: UploadFile = File(...),
    client_id: int = Form(...),
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Predict from uploaded image + Grad-CAM + LIME explanations.

    This endpoint proxies the request to the Colab inference orchestrator (via ngrok)
    where the xAI computation happens on cloud GPUs.

    Args:
        file: Uploaded blood cell microscopy image
        client_id: Database ID of the client (used to fetch client_name)
        supabase: Supabase client for database access

    Returns:
        JSON with prediction, confidence, and xAI visualizations (base64 encoded images)

    Note:
        - The INFERENCE_ORCHESTRATOR_URL must be updated in .env when starting a new Colab session
        - For local xAI computation, use the /upload-xai-local endpoint instead
    """
    try:
        logger.info(f"[PROXY] xAI inference request for client_id: {client_id}, file: {file.filename}")

        # Fetch client from database to get client_name
        client_response = supabase.from_("clients").select("client_name").eq("id", client_id).execute()

        if not client_response.data or len(client_response.data) == 0:
            logger.error(f"[PROXY] Client with id {client_id} not found in database")
            raise HTTPException(
                status_code=404,
                detail=f"Client with id {client_id} not found"
            )

        client_name = client_response.data[0]["client_name"]
        logger.info(f"[PROXY] Fetched client_name: {client_name}")
        logger.info(f"[PROXY] Forwarding to orchestrator: {INFERENCE_ORCHESTRATOR_URL}")

        # Read file content
        file_content = await file.read()

        # Prepare multipart form data (file only)
        files = {
            "file": (file.filename, file_content, file.content_type)
        }

        # Send client_name as query parameter
        params = {
            "client_name": client_name
        }

        # Forward request to Colab orchestrator
        response = requests.post(
            f"{INFERENCE_ORCHESTRATOR_URL}/upload-xai",
            files=files,
            params=params,  # client_name as query param
            timeout=600  # xAI can take time (LIME is expensive)
        )

        response.raise_for_status()
        result = response.json()

        logger.info(f"[PROXY] Inference successful. Prediction: {result.get('prediction')}")
        return JSONResponse(status_code=200, content=result)

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        logger.error(f"[PROXY] Request timeout to orchestrator")
        raise HTTPException(
            status_code=504,
            detail="Inference request timed out. xAI generation takes 10-15 seconds."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"[PROXY] Orchestrator error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Orchestrator API Error: {e.response.text}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to inference orchestrator: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[PROXY] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))