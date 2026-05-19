from pydantic import BaseModel, Field
from typing import Optional, List


class DetectionRequest(BaseModel):
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


# NEW — for URL-based requests
class URLDetectionRequest(BaseModel):
    url: str = Field(
        ...,
        description="URL of the news article to fetch and analyze"
    )


class ModelPrediction(BaseModel):
    model_name: str
    prediction: str
    confidence: float


class DetectionResponse(BaseModel):
    verdict: str
    confidence_score: float
    model_predictions: List[ModelPrediction]
    reasons: List[str]
    fact_check_results: List[str]
    processing_time_seconds: float
    article_title: Optional[str] = None      # NEW — shown in frontend
    article_authors: Optional[List[str]] = None  # NEW


class HealthResponse(BaseModel):
    status: str
    version: str
    app_name: str