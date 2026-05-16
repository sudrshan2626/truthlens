# ml/eda.py
# Run this to visualize and understand the dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_PATH = "data/processed/dataset.csv"
CHARTS_DIR     = "data/charts"

os.makedirs(CHARTS_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(PROCESSED_PATH)

# ── 1. Basic Stats ─────────────────────────────────────────────────────────
print("\n── Dataset Info ─────────────────────────────")
print(f"Total rows    : {len(df)}")
print(f"Columns       : {list(df.columns)}")
print(f"Label counts  :\n{df['label'].value_counts()}")
print(f"Null values   :\n{df.isnull().sum()}")

# ── 2. Text Length Analysis ────────────────────────────────────────────────
df["text_length"] = df["content"].apply(lambda x: len(str(x).split()))

print("\n── Text Length Stats ────────────────────────")
print(df.groupby("label")["text_length"].describe())

# ── 3. Label Distribution Chart ───────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(
    x="label",
    data=df,
    palette=["#e74c3c", "#2ecc71"]
)
plt.title("Label Distribution (0=FAKE, 1=REAL)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks([0, 1], ["FAKE", "REAL"])
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/label_distribution.png")
print(f"\n✅ Saved: {CHARTS_DIR}/label_distribution.png")

# ── 4. Text Length Distribution Chart ─────────────────────────────────────
plt.figure(figsize=(10, 4))
for label, color, name in [(0, "#e74c3c", "FAKE"), (1, "#2ecc71", "REAL")]:
    subset = df[df["label"] == label]["text_length"]
    sns.histplot(
        subset,
        bins=50,
        color=color,
        label=name,
        alpha=0.6,
        kde=True
    )
plt.title("Text Length Distribution by Label")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/text_length_distribution.png")
print(f"✅ Saved: {CHARTS_DIR}/text_length_distribution.png")

print("\n── Sample FAKE rows ─────────────────────────")
print(df[df["label"] == 0]["content"].head(3).to_string())

print("\n── Sample REAL rows ─────────────────────────")
print(df[df["label"] == 1]["content"].head(3).to_string())