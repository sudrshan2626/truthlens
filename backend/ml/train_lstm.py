# ml/train_lstm.py
# Trains an LSTM deep learning model for fake news detection.
# Run: python -m ml.train_lstm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import re
from sklearn.model_selection import train_test_split
from app.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = "data/processed/dataset.csv"
MODELS_DIR   = "ml/trained_models"
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_VOCAB    = 20000   # top 20k words
MAX_LEN      = 300     # truncate/pad to 300 tokens
EMBED_DIM    = 128     # embedding vector size
HIDDEN_DIM   = 256     # LSTM hidden state size
BATCH_SIZE   = 64
EPOCHS       = 5
LEARNING_RATE= 0.001


# ── 1. Build Vocabulary ───────────────────────────────────────────────────────

def build_vocab(texts, max_vocab=MAX_VOCAB):
    """
    Builds a word → integer index mapping from training texts.
    
    Special tokens:
    - <PAD> = 0  (padding shorter sequences)
    - <UNK> = 1  (unknown words not in vocabulary)
    """
    from collections import Counter

    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # Take top max_vocab words by frequency
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    logger.info(f"Vocabulary built: {len(vocab)} tokens")
    return vocab


def encode_text(text, vocab, max_len=MAX_LEN):
    """Converts text string to padded list of integer indices."""
    tokens = text.split()[:max_len]
    encoded = [vocab.get(token, 1) for token in tokens]  # 1 = <UNK>

    # Pad with zeros to reach max_len
    padded = encoded + [0] * (max_len - len(encoded))
    return padded


# ── 2. PyTorch Dataset ────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    """
    Custom PyTorch Dataset.
    PyTorch requires __len__ and __getitem__ methods.
    """

    def __init__(self, texts, labels, vocab):
        self.texts  = [encode_text(t, vocab) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx],  dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


# ── 3. LSTM Model ─────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.
    
    Architecture:
    Input text → Embedding → LSTM → Dropout → Linear → Sigmoid → 0/1
    
    Why LSTM over plain RNN?
    LSTMs have a 'memory cell' that can learn long-range dependencies.
    They solve the vanishing gradient problem of plain RNNs.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()

        # Embedding layer: maps word index → dense vector
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0       # <PAD> token produces zero vector
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,           # 2 stacked LSTM layers
            batch_first=True,       # input shape: (batch, seq, features)
            dropout=0.3,            # dropout between LSTM layers
            bidirectional=True      # reads sequence forward AND backward
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Final classification layer
        # bidirectional = hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, 1)

        # Sigmoid squashes output to [0, 1] probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)

        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Use last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        # hidden shape: (batch_size, hidden_dim * 2)

        dropped = self.dropout(hidden)
        output  = self.fc(dropped)
        prob    = self.sigmoid(output)

        return prob.squeeze(1)


# ── 4. Training Loop ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    """Runs one training epoch. Returns average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for texts, labels in loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # clear previous gradients
        predictions = model(texts)      # forward pass
        loss = criterion(predictions, labels)
        loss.backward()                 # backpropagation
        optimizer.step()               # update weights

        total_loss += loss.item()
        predicted_labels = (predictions > 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, criterion):
    """Evaluates model on validation/test data. Returns loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():               # no gradient calculation needed
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            predictions   = model(texts)
            loss          = criterion(predictions, labels)

            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    X  = df["content"].astype(str).tolist()
    y  = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build vocabulary from training data only
    vocab = build_vocab(X_train)
    joblib.dump(vocab, f"{MODELS_DIR}/lstm_vocab.pkl")
    logger.info("✅ Vocabulary saved")

    # Create datasets and loaders
    train_dataset = FakeNewsDataset(X_train, y_train, vocab)
    test_dataset  = FakeNewsDataset(X_test,  y_test,  vocab)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # Build model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()                          # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_accuracy = 0.0

    print(f"\n{'='*60}")
    print(f"  Training LSTM — {EPOCHS} epochs on {DEVICE}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate_epoch(model, test_loader, criterion)

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_model.pt")
            logger.info(f"  ✅ New best model saved (acc: {best_accuracy:.4f})")

    print(f"\n  🏆 Best Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"  Model saved to: {MODELS_DIR}/lstm_model.pt")


if __name__ == "__main__":
    main()