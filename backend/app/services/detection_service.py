# app/services/detection_service.py

import time
from app.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    ModelPrediction
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionService:
    """
    Core business logic for fake news detection.
    Stubs will be replaced phase by phase.
    """

    def __init__(self):
        logger.info("DetectionService initialized")

    async def analyze(self, request: DetectionRequest) -> DetectionResponse:
        """
        Main analysis pipeline.
        """
        start_time = time.time()
        logger.info(f"Analyzing text of length: {len(request.text)}")

        # Stub model prediction — replaced in Phase 4
        model_predictions = [
            ModelPrediction(
                model_name="placeholder",
                prediction="UNCERTAIN",
                confidence=0.5
            )
        ]

        # Stub reasons — replaced in Phase 7
        reasons = [
            "Analysis pipeline not yet connected.",
            "ML models will be integrated in Phase 4.",
            "RAG pipeline will be integrated in Phase 5.",
            "Fact-check API will be integrated in Phase 6.",
            "Gemini explanations will be integrated in Phase 7."
        ]

        processing_time = round(time.time() - start_time, 3)

        return DetectionResponse(
            verdict="UNCERTAIN",
            confidence_score=0.5,
            model_predictions=model_predictions,
            reasons=reasons,
            fact_check_results=[],
            processing_time_seconds=processing_time
        )