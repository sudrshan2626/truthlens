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
    Same cleaning applied during training.
    CRITICAL: Must match ml/build_dataset.py clean_text() exactly.
    If cleaning differs between training and inference → model performance drops.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DetectionService:
    """
    Orchestrates the full detection pipeline.
    Imports predictor lazily to avoid loading models at import time.
    """

    def __init__(self):
        self._predictor = None
        logger.info("DetectionService initialized")

    @property
    def predictor(self):
        """
        Lazy loading — models load only on first request, not at startup.
        This keeps startup time fast.
        """
        if self._predictor is None:
            from ml.predictor import get_predictor
            self._predictor = get_predictor()
        return self._predictor

    async def analyze(self, request: DetectionRequest) -> DetectionResponse:
        """
        Full analysis pipeline:
        1. Clean text
        2. Run all ML models
        3. Build response (Gemini reasons added in Phase 6)
        """
        start_time = time.time()
        logger.info(f"Analyzing text — length: {len(request.text)} chars")

        # Step 1: Clean input text
        cleaned_text = clean_text(request.text)
        logger.info(f"Cleaned text length: {len(cleaned_text)} chars")

        # Step 2: Run ML ensemble
        prediction = self.predictor.predict(cleaned_text)
        logger.info(
            f"Verdict: {prediction['verdict']} | "
            f"Confidence: {prediction['confidence']}"
        )

        # Step 3: Build ModelPrediction schema objects
        model_predictions = [
            ModelPrediction(
                model_name=r["model_name"],
                prediction=r["prediction"],
                confidence=r["confidence"]
            )
            for r in prediction["model_results"]
        ]

        # Step 4: Stub reasons — replaced in Phase 6 with Gemini
        reasons = [
            f"Ensemble verdict: {prediction['verdict']} with {prediction['confidence']*100:.1f}% confidence.",
            f"Logistic Regression: {model_predictions[0].prediction} ({model_predictions[0].confidence*100:.1f}%)",
            f"Passive Aggressive: {model_predictions[1].prediction} ({model_predictions[1].confidence*100:.1f}%)",
            f"Random Forest: {model_predictions[2].prediction} ({model_predictions[2].confidence*100:.1f}%)",
            f"LSTM: {model_predictions[3].prediction} ({model_predictions[3].confidence*100:.1f}%)",
        ]

        processing_time = round(time.time() - start_time, 3)
        logger.info(f"Analysis complete in {processing_time}s")

        return DetectionResponse(
            verdict=prediction["verdict"],
            confidence_score=prediction["confidence"],
            model_predictions=model_predictions,
            reasons=reasons,
            fact_check_results=[],
            processing_time_seconds=processing_time
        )