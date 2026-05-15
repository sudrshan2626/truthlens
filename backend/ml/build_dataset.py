# ml/build_dataset.py

import pandas as pd
import os
import re
from datasets import load_dataset
from app.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH   = f"{PROCESSED_DIR}/dataset.csv"


def load_fakenews_dataset() -> pd.DataFrame:
    """
    Downloads the Fake News dataset from HuggingFace.
    Source: GonzaloA/fake_news (~44,000 rows)
    Label: 0 = FAKE, 1 = REAL
    """
    logger.info("Downloading Fake News dataset from HuggingFace...")

    try:
        dataset = load_dataset("GonzaloA/fake_news", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load fake news dataset: {e}")
        return pd.DataFrame()

    rows = []
    for split in ["train", "test"]:
        if split not in dataset:
            continue
        for item in dataset[split]:
            # Label in this dataset: 0 = FAKE, 1 = REAL
            rows.append({
                "content": str(item.get("text", "") or item.get("title", "")),
                "label": int(item["label"])
            })

    df = pd.DataFrame(rows)
    logger.info(f"Fake News dataset loaded: {len(df)} rows")
    return df


def load_liar_dataset() -> pd.DataFrame:
    """
    Downloads the LIAR dataset from HuggingFace.
    Maps 6-class labels to binary: 0 = FAKE, 1 = REAL
    """
    logger.info("Downloading LIAR dataset from HuggingFace...")

    try:
        dataset = load_dataset("liar", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load LIAR dataset: {e}")
        return pd.DataFrame()

    # LIAR 6 classes → binary
    # 0=pants-fire, 1=false, 2=barely-true → FAKE
    # 3=half-true, 4=mostly-true, 5=true   → REAL
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    rows = []
    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue
        for item in dataset[split]:
            rows.append({
                "content": str(item["statement"]),
                "label": label_map.get(int(item["label"]), 0)
            })

    df = pd.DataFrame(rows)
    logger.info(f"LIAR dataset loaded: {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    """
    Cleans a single text string.
    Steps: lowercase → remove URLs → remove special chars → strip whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_dataset():
    """
    Main pipeline: load → merge → clean → balance → save
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    fakenews_df = load_fakenews_dataset()
    liar_df     = load_liar_dataset()

    # ── Step 2: Merge ─────────────────────────────────────────────────────────
    frames = [df for df in [fakenews_df, liar_df] if not df.empty]

    if not frames:
        logger.error("No datasets loaded. Check your internet connection.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows after merge: {len(df)}")

    # ── Step 3: Drop duplicates ───────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["content"])
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    # ── Step 4: Drop nulls ────────────────────────────────────────────────────
    df = df.dropna(subset=["content", "label"])

    # ── Step 5: Clean text ────────────────────────────────────────────────────
    logger.info("Cleaning text... (this takes 1-2 minutes)")
    df["content"] = df["content"].apply(clean_text)

    # Drop rows too short after cleaning
    df = df[df["content"].str.len() > 20]

    # ── Step 6: Balance classes ───────────────────────────────────────────────
    fake_df = df[df["label"] == 0]
    real_df = df[df["label"] == 1]

    min_count = min(len(fake_df), len(real_df))
    logger.info(f"Balancing → FAKE: {len(fake_df)}, REAL: {len(real_df)}, using {min_count} each")

    fake_df = fake_df.sample(n=min_count, random_state=42)
    real_df = real_df.sample(n=min_count, random_state=42)

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # ── Step 7: Shuffle ───────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Step 8: Save ──────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"✅ Dataset saved → {OUTPUT_PATH}")
    logger.info(f"   Total rows : {len(df)}")
    logger.info(f"   FAKE rows  : {len(df[df['label'] == 0])}")
    logger.info(f"   REAL rows  : {len(df[df['label'] == 1])}")

    return df


if __name__ == "__main__":
    df = build_dataset()

    if not df.empty:
        print("\n── Label Distribution ───────────────────────")
        print(df["label"].value_counts())
        print(f"\n── Shape: {df.shape}")
        print("\n── Sample rows ──────────────────────────────")
        print(df.head(3).to_string())