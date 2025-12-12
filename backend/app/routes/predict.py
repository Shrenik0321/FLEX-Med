import io
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.schemas.prediction import PredictionResponse
from app.services.model_service import predict_image, load_model
from app.config import get_settings

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)


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

        # 4. Return response in the same format as /inference
        return JSONResponse(
            status_code=200,
            content={
                "prediction": result["prediction"],
                "label": result.get("label", result["prediction"]),  # Backward compatibility
                "confidence": result["confidence"],
                "class": result["class"],
                "model": result["model"],
                "model_path": result["model_path"],
                "device": result["device"],
                "all_probabilities": result["all_probabilities"],
            }
        )

    except HTTPException as he:
        raise he  # Pass through standard HTTP errors
    except Exception as e:
        # Catch unforeseen errors (variables missing, logic bugs)
        error_msg = f"Unexpected Server Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg) from e
