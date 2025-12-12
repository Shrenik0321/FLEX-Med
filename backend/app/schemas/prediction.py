from typing import Optional, Dict

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label")
    label: Optional[str] = Field(None, description="Predicted class label (backward compatibility)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability for the predicted class")
    class_index: int = Field(..., alias="class", description="Class index of the prediction")
    model: str = Field(..., description="Model name/architecture used")
    model_path: str = Field(..., description="Path to the model weights used")
    device: str = Field(..., description="Device used for inference")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    true_label: Optional[str] = Field(None, description="Ground truth label (when available)")
    sample_index: Optional[int] = Field(None, description="Dataset index (when using random sampling)")
    
    class Config:
        populate_by_name = True

