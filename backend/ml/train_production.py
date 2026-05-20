# ml/train_production.py
# Trains small, fast models optimized for deployment.
# Output files are committed to GitHub (< 20MB total).

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from app.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = "data/processed/dataset.csv"
PROD_DIR     = "ml/prod_models"


def main():
    os.makedirs(PROD_DIR, exist_ok=True)

    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    X  = df["content"].astype(str)
    y  = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Lightweight TF-IDF (30k features instead of 50k)
    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Logistic Regression
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred   = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    # Save
    joblib.dump(vectorizer, f"{PROD_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(model,      f"{PROD_DIR}/logistic_regression.pkl")
    logger.info(f"✅ Production models saved to {PROD_DIR}/")

    # Check sizes
    for f in os.listdir(PROD_DIR):
        size = os.path.getsize(f"{PROD_DIR}/{f}") / (1024 * 1024)
        logger.info(f"   {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
    