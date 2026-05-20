import time
import re
from app.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    ModelPrediction
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Must match ml/build_dataset.py clean_text() exactly.
    Training-serving consistency is critical.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DetectionService:

    def __init__(self):
        self._predictor      = None
        self._gemini_service = None
        logger.info("DetectionService initialized")

    @property
    def predictor(self):
        """
        Loads correct predictor based on environment.
        - Production (Render): uses lightweight prod_predictor
        - Development (local): uses full predictor with all 4 models
        """
        if self._predictor is None:
            import os
            env = os.getenv("ENVIRONMENT", "development")

            if env == "production":
                from ml.prod_predictor import get_production_predictor
                self._predictor = get_production_predictor()
            else:
                from ml.predictor import get_predictor
                self._predictor = get_predictor()

        return self._predictor

    @property
    def gemini(self):
        """Lazy-loads Gemini service on first request."""
        if self._gemini_service is None:
            from app.services.gemini_service import get_gemini_service
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    async def analyze(self, request: DetectionRequest) -> DetectionResponse:
        """
        Full pipeline:
        1. Clean text
        2. ML ensemble prediction
        3. Gemini generates 5 reasons
        4. Return structured response
        """
        start_time = time.time()
        logger.info(f"Analyzing text — {len(request.text)} chars")

        # Step 1: Clean
        cleaned_text = clean_text(request.text)

        # Step 2: ML Prediction
        prediction = self.predictor.predict(cleaned_text)
        logger.info(
            f"Verdict: {prediction['verdict']} | "
            f"Confidence: {prediction['confidence']}"
        )

        # Step 3: Build ModelPrediction objects
        model_predictions = [
            ModelPrediction(
                model_name=r["model_name"],
                prediction=r["prediction"],
                confidence=r["confidence"]
            )
            for r in prediction["model_results"]
        ]

        # Step 4: Gemini reasons (async call)
        reasons = await self.gemini.generate_reasons(
            text=request.text,              # send original (not cleaned) for Gemini
            verdict=prediction["verdict"],
            confidence=prediction["confidence"],
            model_results=prediction["model_results"]
        )

        processing_time = round(time.time() - start_time, 3)
        logger.info(f"Total analysis time: {processing_time}s")

        return DetectionResponse(
            verdict=prediction["verdict"],
            confidence_score=prediction["confidence"],
            model_predictions=model_predictions,
            reasons=reasons,
            fact_check_results=[],          # Phase 7 will fill this
            processing_time_seconds=processing_time
        )