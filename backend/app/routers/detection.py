from fastapi import APIRouter, HTTPException 
from app.schemas.detection import DetectionRequest, DetectionResponse
from app.services.detection_service import DetectionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# APIRouter is like a mini FastAPI app - groups related endpoints
router = APIRouter(
    prefix="/api",
    tags=["Detection"]
)

# Create one service instance
detection_service = DetectionService()

@router.post("/detect", response_model=DetectionResponse)
async def detect_fake_news(request: DetectionRequest):
    """
    Analyze a news article or claim for authenticity.

    **text**: The article or claim to analyze
    **source_url**: Original URL of the article 

    Returns verdict, confidence score, model predictions, and AI-generated reasons.
    """
    try:
        logger.info("Received detection request")
        result = await detection_service.analyze(request)
        return result
    
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )