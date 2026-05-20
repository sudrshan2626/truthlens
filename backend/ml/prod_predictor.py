import joblib
import numpy as np
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROD_DIR = "ml/prod_models"


class ProductionPredictor:
    """
    Lightweight predictor for deployment.
    Uses only TF-IDF + Logistic Regression.
    Fast, small memory footprint, accurate.
    """

    def __init__(self):
        logger.info("Loading production models...")
        self.vectorizer = joblib.load(f"{PROD_DIR}/tfidf_vectorizer.pkl")
        self.lr_model   = joblib.load(f"{PROD_DIR}/logistic_regression.pkl")
        logger.info("✅ Production models loaded")

    def predict(self, text: str) -> dict:
        vec        = self.vectorizer.transform([text])
        label      = int(self.lr_model.predict(vec)[0])
        confidence = float(self.lr_model.predict_proba(vec)[0][label])

        prediction = "REAL" if label == 1 else "FAKE"

        return {
            "verdict":    prediction,
            "confidence": round(confidence, 4),
            "model_results": [
                {
                    "model_name": "Logistic Regression",
                    "prediction": prediction,
                    "confidence": round(confidence, 4)
                }
            ]
        }


_instance = None

def get_production_predictor() -> ProductionPredictor:
    global _instance
    if _instance is None:
        _instance = ProductionPredictor()
    return _instance