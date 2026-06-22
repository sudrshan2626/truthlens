# 🔍 TruthLens — Fake News Detector

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-4285F4?style=flat&logo=google&logoColor=white)](https://deepmind.google/gemini)

A full-stack application that classifies news articles as Real or Fake by combining TF-IDF-based classical ML models with Gemini AI-driven explanations.

I built TruthLens to explore how traditional Machine Learning speed can be combined with LLM interpretability to make fake news detection both accurate and explainable — not just a black box verdict.

**[🚀 Live Demo](https://your-vercel-url.vercel.app) · [📖 API Docs](https://truthlens-api-mijs.onrender.com/docs)**

---

## What It Does

A user pastes a news article or drops a URL. TruthLens:
1. Cleans and vectorizes the text using TF-IDF
2. Runs it through a Logistic Regression classifier trained on 29,736 labeled samples
3. Sends the verdict and article to Gemini AI via LangChain
4. Returns a structured response: verdict, confidence score, and 5 specific reasons

---

## Features

- **Hybrid classification** — Scikit-learn handles initial classification; Gemini AI generates a human-readable explanation of why the article was flagged
- **97.58% test accuracy** — Achieved on a balanced FAKE/REAL dataset; focused on minimizing false positives by combining ML classification with LLM reasoning
- **URL analysis** — Uses newspaper3k to scrape and extract article text directly from a news link
- **Two separate requirements files** — Discovered during deployment that Mac-generated `requirements.txt` included PyTorch and CUDA packages that conflicted on Render's Linux environment; solved by maintaining `requirements-prod.txt` with only what the API actually needs
- **REST API with auto-generated docs** — FastAPI's `/docs` endpoint made it easy to test and share the API during development

---

## Architecture
┌─────────────────────────────────────────────────────┐

│              React Frontend (Vercel)                 │

│         Paste Text / News URL → Get Verdict          │

└─────────────────────┬───────────────────────────────┘

│ REST API

┌─────────────────────▼───────────────────────────────┐

│              FastAPI Backend (Render)                 │

│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │

│  │  ML Models   │  │  LangChain  │  │  Gemini AI │  │

│  │  TF-IDF +    │  │  Prompt     │  │  5 Reasons │  │

│  │  Logistic    │  │  Templates  │  │  Generator │  │

│  │  Regression  │  │             │  │            │  │

│  └──────────────┘  └─────────────┘  └────────────┘  │

│  ┌──────────────┐                                    │

│  │  newspaper3k │  ← URL article scraper             │

│  └──────────────┘                                    │

└─────────────────────────────────────────────────────┘

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 + Vite | UI framework |
| Styling | Tailwind CSS | Utility-first CSS |
| Backend | FastAPI + Python 3.12 | REST API |
| ML | scikit-learn + TF-IDF | Text classification |
| LLM | LangChain + Gemini 2.0 Flash | Explanation generation |
| Article Scraping | newspaper3k | URL content extraction |
| Deployment | Render + Vercel | Cloud hosting |

---

## Model Performance

| Model | Accuracy | Training Set |
|-------|---------|-------------|
| TF-IDF + Logistic Regression | **97.58%** | 23,788 samples |

Test set: 5,948 samples (80/20 stratified split)

**Data sources:**
- [LIAR dataset](https://huggingface.co/datasets/liar) — 12,791 political statements with truthfulness labels
- [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) — 44,898 news articles labeled FAKE/REAL

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Gemini API key ([free at aistudio.google.com](https://aistudio.google.com/apikey))

### Backend

```bash
git clone https://github.com/YOUR_USERNAME/truthlens.git
cd truthlens/backend

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env

# First-time setup
python -m ml.build_dataset
python -m ml.train_classical
python -m ml.train_lstm

uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API Reference

### POST /api/detect

```json
// Request
{
  "text": "News article or claim text here"
}

// Response
{
  "verdict": "FAKE",
  "confidence_score": 0.97,
  "model_predictions": [
    {"model_name": "Logistic Regression", "prediction": "FAKE", "confidence": 0.97}
  ],
  "reasons": [
    "**Sensationalism**: The use of all-caps and extreme language...",
    "**Source Attribution**: No named sources or verifiable citations..."
  ],
  "processing_time_seconds": 2.3
}
```

### POST /api/detect-url

```json
{"url": "https://news-site.com/article"}
```


---

## Project Structure
truthlens/

├── backend/

│   ├── app/

│   │   ├── main.py              # FastAPI app entry point

│   │   ├── config.py            # Pydantic settings from .env

│   │   ├── routers/             # detection.py endpoints

│   │   ├── services/            # detection, gemini, url services

│   │   ├── schemas/             # Request/response Pydantic models

│   │   └── utils/               # Logger

│   ├── ml/

│   │   ├── build_dataset.py     # HuggingFace download + cleaning

│   │   ├── train_classical.py   # TF-IDF + LR/RF/PA training

│   │   ├── train_lstm.py        # PyTorch LSTM training

│   │   ├── train_production.py  # Lightweight model for deployment

│   │   ├── predictor.py         # Full 4-model inference

│   │   └── prod_predictor.py    # Production inference (LR only)

│   ├── requirements.txt         # Full dev dependencies

│   └── requirements-prod.txt    # Minimal deployment dependencies

└── frontend/

├── src/

│   ├── App.jsx              # Main layout and state

│   ├── api/detection.js     # Axios API calls

│   └── components/          # VerdictBadge, ModelTable, etc.

└── vite.config.js           # Proxy config for local dev

---

## Environment Variables

```bash
# backend/.env
GEMINI_API_KEY=your_key_here
ENVIRONMENT=development
FRONTEND_URL=http://localhost:5173

# frontend/.env.production
VITE_API_URL=https://your-render-url.onrender.com
```

---

## Challenges and How I Solved Them

**Deployment dependency conflicts** — `pip freeze` on Mac captured PyTorch, CUDA drivers, and ARM-specific binaries that don't exist on Render's Linux environment. Fixed by writing a separate `requirements-prod.txt` with only the packages the API actually calls at runtime.

**PassiveAggressiveClassifier has no predict_proba** — Discovered during inference that this classifier uses margin-based scoring, not probability estimates. Fixed by using `decision_function()` and converting the raw score to a [0,1] range via sigmoid.

**LangChain import paths changed in v0.2** — `langchain.prompts` and `langchain.output_parsers` were removed. Fixed by migrating imports to `langchain_core`.

**Training vs production model size** — Random Forest and LSTM models were too large (80MB+) for Render's free tier. Solved by training a lightweight production variant (TF-IDF + LR, 1.3MB total) that achieves the same 97.58% accuracy.

---

## Roadmap

- [ ] Add BERT/RoBERTa for deeper semantic understanding
- [ ] Store analysis history per session
- [ ] Add confidence threshold controls in the UI
- [ ] Expand fact-check integration with Google Fact Check API
- [ ] Add user authentication and saved analyses

---

## References

- [LIAR: A Benchmark Dataset for Fake News Detection](https://arxiv.org/abs/1705.00648) — Wang, 2017
- [LangChain Documentation](https://docs.langchain.com)
- [Google Gemini API](https://ai.google.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

