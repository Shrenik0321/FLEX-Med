import io
import logging
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.schemas.prediction import PredictionResponse
from app.config import get_settings
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
# PREDICTION + XAI (GRAD-CAM)
# ============================================================
@router.post("/upload-xai")
async def upload_xai(file: UploadFile = File(...)):
    """
    Predict from uploaded image + Grad-CAM explanation
    """
    try:
        settings = get_settings()

        # ----------------------------------------------------
        # Load cached model bundle (NO reloading)
        # ----------------------------------------------------
        bundle = load_model(settings)
        model = bundle.model
        device = bundle.device

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ----------------------------------------------------
        # Preprocess (same as training)
        # ----------------------------------------------------
        input_tensor = COMMON_TRANSFORM(image).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------
        target_layers = get_gradcam_target_layers(model, bundle.model_name)

        gradcam_base64 = generate_gradcam_base64(
            model=model,
            input_tensor=input_tensor,
            target_layers=target_layers,
            class_idx=pred_idx,
        )

        lime_base64 = generate_lime_base64(
            model=model,
            input_tensor=input_tensor,
            class_idx=pred_idx,
            device=device,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------
        return JSONResponse(
            status_code=200,
            content={
                "prediction": bundle.class_names[pred_idx],
                "confidence": confidence,
                "class": pred_idx,
                "model": bundle.model_name,
                "model_path": str(settings.model_path),
                "device": str(bundle.device),
                "xai": {
                    "gradcam": {
                        "image_base64": gradcam_base64
                    },
                    "lime": {
                        "image_base64": lime_base64
                    }
                },
            },
        )

    except Exception as e:
        logger.error("upload-xai failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
