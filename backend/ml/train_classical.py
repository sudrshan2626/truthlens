
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH  = "data/processed/dataset.csv"
MODELS_DIR    = "ml/trained_models"


def load_data():
    """Loads and splits dataset into train/test sets."""
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    X = df["content"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 80% train, 20% test
        random_state=42,     # reproducible split
        stratify=y           # keep class balance in both sets
    )

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_tfidf_vectorizer(X_train):
    """
    Fits TF-IDF vectorizer on training data.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - TF: how often a word appears in THIS document
    - IDF: how rare the word is across ALL documents
    - Result: common words (the, a, is) get low scores
              rare but meaningful words get high scores
    """
    logger.info("Fitting TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=50000,    # top 50,000 words by frequency
        ngram_range=(1, 2),    # single words AND two-word phrases
        min_df=2,              # ignore words appearing in < 2 docs
        max_df=0.95,           # ignore words in > 95% of docs (too common)
        sublinear_tf=True      # use log(TF) to dampen extreme frequencies
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Save vectorizer — MUST use same vectorizer at prediction time
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    logger.info("✅ TF-IDF vectorizer saved")

    return vectorizer, X_train_vec


def evaluate_model(model_name, model, X_test_vec, y_test):
    """Evaluates a trained model and prints detailed metrics."""
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def train_logistic_regression(X_train_vec, y_train):
    """
    Logistic Regression — strong baseline for text classification.
    
    Why: Despite the name, it's a classification algorithm.
    Finds a linear boundary in TF-IDF feature space.
    Fast, interpretable, rarely overfits on text data.
    """
    logger.info("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,      # enough iterations to converge
        C=1.0,              # regularization strength (1.0 = default)
        solver="lbfgs",     # optimizer — efficient for medium datasets
        n_jobs=-1           # use all CPU cores
    )
    model.fit(X_train_vec, y_train)

    joblib.dump(model, f"{MODELS_DIR}/logistic_regression.pkl")
    logger.info("✅ Logistic Regression saved")

    return model


def train_passive_aggressive(X_train_vec, y_train):
    """
    Passive Aggressive Classifier — designed for text streams.
    
    Why: Updates only on mistakes (passive on correct, aggressive on wrong).
    Excellent for large text datasets, very fast training.
    """
    logger.info("Training Passive Aggressive Classifier...")

    model = PassiveAggressiveClassifier(
        C=0.5,              # aggressiveness — lower = more conservative
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    joblib.dump(model, f"{MODELS_DIR}/passive_aggressive.pkl")
    logger.info("✅ Passive Aggressive saved")

    return model


def train_random_forest(X_train_vec, y_train):
    """
    Random Forest — ensemble of decision trees.
    
    Why: Handles non-linear patterns TF-IDF + linear models miss.
    Robust to noise. Slower than the above but often more accurate.
    Note: We use fewer trees (100) to keep training fast.
    """
    logger.info("Training Random Forest... (this takes 2-3 minutes)")

    model = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=20,       # prevent overfitting
        random_state=42,
        n_jobs=-1           # parallel training on all cores
    )
    model.fit(X_train_vec, y_train)

    joblib.dump(model, f"{MODELS_DIR}/random_forest.pkl")
    logger.info("✅ Random Forest saved")

    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and split data
    X_train, X_test, y_train, y_test = load_data()

    # Build TF-IDF features
    vectorizer, X_train_vec = build_tfidf_vectorizer(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train all three models
    lr_model  = train_logistic_regression(X_train_vec, y_train)
    pa_model  = train_passive_aggressive(X_train_vec, y_train)
    rf_model  = train_random_forest(X_train_vec, y_train)

    # Evaluate all three
    results = {}
    results["Logistic Regression"]      = evaluate_model("Logistic Regression",      lr_model,  X_test_vec, y_test)
    results["Passive Aggressive"]        = evaluate_model("Passive Aggressive",        pa_model,  X_test_vec, y_test)
    results["Random Forest"]             = evaluate_model("Random Forest",             rf_model,  X_test_vec, y_test)

    # Summary
    print(f"\n{'='*50}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(acc * 40)
        print(f"  {name:<28} {acc:.4f}  {bar}")

    best = max(results, key=results.get)
    print(f"\n  🏆 Best model: {best} ({results[best]*100:.2f}%)")


if __name__ == "__main__":
    main()