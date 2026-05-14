# app/schemas/detection.py

from pydantic import BaseModel, Field
from typing import Optional, List


class DetectionRequest(BaseModel):
    """
    Schema for incoming detection request from frontend.
    """
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The news article or claim to be analyzed"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Optional URL of the original article"
    )


class ModelPrediction(BaseModel):
    """
    Schema for a single ML model's prediction result.
    """
    model_name: str
    prediction: str       # "REAL" or "FAKE"
    confidence: float     # 0.0 to 1.0


class DetectionResponse(BaseModel):
    """
    Schema for the full response sent back to frontend.
    """
    verdict: str
    confidence_score: float
    model_predictions: List[ModelPrediction]
    reasons: List[str]
    fact_check_results: List[str]
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """
    Schema for health check endpoint.
    """
    status: str
    version: str
    app_name: str