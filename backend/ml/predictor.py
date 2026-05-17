import joblib
import torch
import torch.nn as nn
import numpy as np
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = "ml/trained_models"
MAX_LEN    = 300
EMBED_DIM  = 128
HIDDEN_DIM = 256
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── LSTM Model Definition ─────────────────────────────────────────────────────
# Must match EXACTLY what was defined in train_lstm.py

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded          = self.embedding(x)
        _, (hidden, _)    = self.lstm(embedded)
        hidden            = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.sigmoid(self.fc(self.dropout(hidden))).squeeze(1)


# ── Helper ────────────────────────────────────────────────────────────────────

def encode_text(text: str, vocab: dict, max_len: int = MAX_LEN) -> list:
    """Converts text to padded list of vocab indices."""
    tokens  = text.lower().split()[:max_len]
    encoded = [vocab.get(token, 1) for token in tokens]
    return encoded + [0] * (max_len - len(encoded))


# ── Model Loader ──────────────────────────────────────────────────────────────

class ModelPredictor:
    """
    Loads all trained models once at startup.
    Provides a single predict() method that returns ensemble results.

    Singleton pattern — only one instance exists across all API requests.
    """

    def __init__(self):
        logger.info("Loading all models into memory...")
        self._load_classical_models()
        self._load_lstm_model()
        logger.info("✅ All models loaded and ready")

    def _load_classical_models(self):
        """Loads TF-IDF vectorizer and three classical ML models."""
        try:
            self.vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
            self.lr_model   = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
            self.pa_model   = joblib.load(f"{MODELS_DIR}/passive_aggressive.pkl")
            self.rf_model   = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
            logger.info("✅ Classical models loaded")
        except Exception as e:
            logger.error(f"Failed to load classical models: {e}")
            raise

    def _load_lstm_model(self):
        """Loads LSTM model and vocabulary."""
        try:
            self.vocab = joblib.load(f"{MODELS_DIR}/lstm_vocab.pkl")

            self.lstm_model = LSTMClassifier(
                vocab_size=len(self.vocab),
                embed_dim=EMBED_DIM,
                hidden_dim=HIDDEN_DIM
            )
            self.lstm_model.load_state_dict(
                torch.load(
                    f"{MODELS_DIR}/lstm_model.pt",
                    map_location=DEVICE,
                    weights_only=True
                )
            )
            self.lstm_model.to(DEVICE)
            self.lstm_model.eval()   # inference mode — disables dropout
            logger.info("✅ LSTM model loaded")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise

    # ── Individual Predictors ─────────────────────────────────────────────────

    def _predict_classical(self, text: str) -> dict:
        """
        Vectorizes text and runs all three classical models.
        Note: PassiveAggressiveClassifier uses decision_function (not predict_proba)
        converted to probability via sigmoid.
        """
        vec = self.vectorizer.transform([text])

        results = {}

        # ── Logistic Regression — supports predict_proba ──────────────────────
        label      = int(self.lr_model.predict(vec)[0])
        confidence = float(self.lr_model.predict_proba(vec)[0][label])
        results["Logistic Regression"] = (label, confidence)

        # ── Passive Aggressive — use decision_function + sigmoid ──────────────
        # decision_function returns a raw score (can be any number)
        # sigmoid maps it to [0, 1] range as a pseudo-probability
        pa_label  = int(self.pa_model.predict(vec)[0])
        pa_score  = float(self.pa_model.decision_function(vec)[0])
        pa_conf   = float(1 / (1 + np.exp(-abs(pa_score))))  # sigmoid of |score|
        results["Passive Aggressive"] = (pa_label, pa_conf)

        # ── Random Forest — supports predict_proba ────────────────────────────
        label      = int(self.rf_model.predict(vec)[0])
        confidence = float(self.rf_model.predict_proba(vec)[0][label])
        results["Random Forest"] = (label, confidence)

        return results

    def _predict_lstm(self, text: str) -> tuple:
        """
        Encodes text and runs LSTM model.
        Returns (label, confidence)
        """
        encoded = encode_text(text, self.vocab)
        tensor  = torch.tensor([encoded], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            prob = self.lstm_model(tensor).item()

        label      = 1 if prob >= 0.5 else 0
        confidence = prob if label == 1 else 1 - prob
        return label, confidence

    # ── Main Predict ──────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Runs all 4 models and returns ensemble result.

        Ensemble logic:
        - Each model casts a vote (FAKE=0, REAL=1)
        - Final verdict = majority vote
        - Confidence = average probability across models
        - If 2 vs 2 tie → UNCERTAIN

        Returns:
            {
                "verdict":      "REAL" | "FAKE" | "UNCERTAIN",
                "confidence":   float (0.0 - 1.0),
                "model_results": list of ModelPrediction dicts
            }
        """
        classical = self._predict_classical(text)
        lstm_label, lstm_conf = self._predict_lstm(text)

        # Build individual model results
        model_results = []

        for model_name, (label, conf) in classical.items():
            model_results.append({
                "model_name": model_name,
                "prediction": "REAL" if label == 1 else "FAKE",
                "confidence": round(conf, 4)
            })

        model_results.append({
            "model_name": "LSTM",
            "prediction": "REAL" if lstm_label == 1 else "FAKE",
            "confidence": round(lstm_conf, 4)
        })

        # Ensemble vote
        votes       = [r["prediction"] for r in model_results]
        real_votes  = votes.count("REAL")
        fake_votes  = votes.count("FAKE")
        all_confs   = [r["confidence"] for r in model_results]
        avg_conf    = round(float(np.mean(all_confs)), 4)

        if real_votes > fake_votes:
            verdict = "REAL"
        elif fake_votes > real_votes:
            verdict = "FAKE"
        else:
            verdict = "UNCERTAIN"   # 2-2 tie

        return {
            "verdict":      verdict,
            "confidence":   avg_conf,
            "model_results": model_results
        }


# ── Singleton Instance ────────────────────────────────────────────────────────
# Instantiated once when module is first imported.
# All API requests reuse this same instance.

_predictor_instance = None

def get_predictor() -> ModelPredictor:
    """Returns the shared ModelPredictor instance (lazy singleton)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ModelPredictor()
    return _predictor_instance