from fastapi import APIRouter, HTTPException
from app.schemas.detection import (
    DetectionRequest,
    URLDetectionRequest,
    DetectionResponse
)
from app.services.detection_service import DetectionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Detection"])
detection_service = DetectionService()


@router.post("/detect", response_model=DetectionResponse)
async def detect_fake_news(request: DetectionRequest):
    """Analyzes pasted text for authenticity."""
    try:
        logger.info("Received text detection request")
        return await detection_service.analyze(request)
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/detect-url", response_model=DetectionResponse)
async def detect_from_url(request: URLDetectionRequest):
    """
    Fetches article from URL then analyzes it.
    Frontend sends a URL → backend scrapes text → runs full pipeline.
    """
    try:
        logger.info(f"Received URL detection request: {request.url}")

        # Step 1: Fetch article text from URL
        from app.services.url_service import get_url_service
        url_service = get_url_service()
        article = url_service.extract_article(request.url)

        if not article["success"]:
            raise HTTPException(
                status_code=422,
                detail=article.get("error", "Failed to extract article")
            )

        # Step 2: Run full detection pipeline on extracted text
        detection_request = DetectionRequest(
            text=article["text"],
            source_url=request.url
        )
        result = await detection_service.analyze(detection_request)

        # Step 3: Attach article metadata to response
        result.article_title   = article.get("title")
        result.article_authors = article.get("authors")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")