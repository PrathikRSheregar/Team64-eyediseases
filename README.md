# Eye Disease Prediction Using AI/ML Algorithms

An end-to-end academic AI/ML platform that classifies eye/retinal images into **Cataract**,
**Diabetic Retinopathy**, **Glaucoma**, or **Normal**, compares nine machine-learning and
deep-learning models, explains its predictions with Grad-CAM, and serves everything through a
FastAPI backend and a React research dashboard.

> **Medical disclaimer:** This project is an AI-assisted **research and educational tool**. It is
> **not a certified medical device** and predictions must never be treated as a diagnosis. Always
> consult a qualified eye-care professional for clinical evaluation. Model performance depends
> entirely on the dataset used to train it and may not generalize to real-world clinical
> populations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [ML Algorithms](#ml-algorithms)
6. [Dataset Structure](#dataset-structure)
7. [Installation](#installation)
8. [Environment Variables](#environment-variables)
9. [Running Locally](#running-locally)
10. [Running with Docker](#running-with-docker)
11. [Training Models](#training-models)
12. [Evaluating Models](#evaluating-models)
13. [Model Comparison & Selection](#model-comparison--selection)
14. [API Documentation](#api-documentation)
15. [Database Structure](#database-structure)
16. [Project Structure](#project-structure)
17. [Testing](#testing)
18. [Screenshots](#screenshots)
19. [Future Improvements](#future-improvements)
20. [Limitations](#limitations)
21. [Medical Disclaimer](#medical-disclaimer)
22. [Authors / Contributors](#authors--contributors)

---

## Project Overview

A user uploads an eye/retinal image through the web app. The backend validates and preprocesses
the image identically to how training data was processed, runs it through the current
**production model** (automatically selected during training as the best-performing model), and
returns:

- Predicted disease class
- Confidence score
- Full probability distribution across all four classes
- The model used
- A Grad-CAM heatmap explanation (for deep-learning models)

Every prediction is stored with its probabilities in PostgreSQL, so users can browse their
history, and admins can view platform-wide statistics and model performance from a research
dashboard.

## Features

- Drag-and-drop image upload with client + server-side validation
- Real-time prediction with confidence and per-class probabilities
- Grad-CAM explainability overlay for deep models; feature importance for traditional ML
- Prediction history with filtering by disease/model/date
- Model comparison dashboard: accuracy, precision, recall, F1, specificity, ROC-AUC, inference
  time, model size, parameter count
- Dataset statistics dashboard: class balance, train/val/test split
- JWT authentication with user/admin roles
- Admin dashboard: total users, total predictions, disease distribution, model usage
- Dark/light mode, responsive design, toasts, skeleton loaders, empty/error states
- Dockerized, environment-variable-driven configuration, Alembic migrations

## Technology Stack

**Frontend:** React, TypeScript, Vite, Tailwind CSS, Axios, React Router, Recharts, react-dropzone

**Backend:** Python, FastAPI, Uvicorn, SQLAlchemy 2.0, PostgreSQL, Pydantic v2, python-jose (JWT),
Passlib/bcrypt

**Machine Learning:** NumPy, Pandas, OpenCV, Pillow, scikit-learn, scikit-image, TensorFlow/Keras,
Matplotlib, Seaborn, Grad-CAM (custom implementation)

**Infrastructure:** Docker, Docker Compose

## System Architecture

```
Eye Image
    │
    ▼
React Frontend (upload, dashboards)
    │  Axios + JWT
    ▼
FastAPI Backend  ──────────────►  PostgreSQL (users, predictions, probabilities, model performance)
    │                                       ▲
    ▼                                       │
Image Preprocessing (shared with training)  │
    │                                       │
    ▼                                       │
Production Model (loaded once at startup) ──┘
    │
    ▼
Prediction + Probabilities + Grad-CAM
```

The `ml/` package is shared between the offline training pipeline and the live backend
(`backend/app/ml/`) so a prediction request is preprocessed **exactly** like training data — this
consistency is what makes the served predictions trustworthy relative to the reported test
metrics.

## ML Algorithms

**Traditional ML** (trained on HOG + color-histogram + statistical features, PCA-compressed,
class-weighted for imbalance):
- Decision Tree
- Random Forest
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Logistic Regression

**Deep Learning / Transfer Learning** (ImageNet-pretrained backbones, frozen head-training phase
followed by fine-tuning of the top backbone layers):
- MobileNetV2
- ResNet50
- EfficientNetB0
- DenseNet121

Adding a new model only requires adding a factory entry in
`ml/models/traditional_models.py` or `ml/models/deep_models.py` and its name to
`ml/config.py::ALL_MODEL_NAMES` — the training, evaluation, comparison, and API layers pick it up
automatically.

## Dataset Structure

Bring your own dataset (none is downloaded or invented). Place images under:

```
dataset/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/
```

If your folders are named differently, edit `CLASS_DIR_MAP` in `ml/config.py` — nothing else in
the pipeline needs to change. You can also point at a dataset stored elsewhere via the
`EYE_DATASET_DIR` environment variable.

The loader validates every image (rejects unreadable/corrupt files), drops exact duplicates by
content hash, and performs a **stratified** train/val/test split (70/15/15 by default) so no image
appears in more than one split.

## Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 16 (or use Docker Compose, see below)
- Docker & Docker Compose (optional but recommended)

### Clone and set up environment variables
```bash
cp .env.example .env
# edit .env - set a real JWT_SECRET_KEY and DB credentials
```

### Backend + ML pipeline dependencies
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r ../ml/requirements.txt
```

### Frontend dependencies
```bash
cd frontend
npm install
cp .env.example .env
```

## Environment Variables

All configuration lives in `.env` (root, consumed by the backend/ML pipeline) and
`frontend/.env` (consumed by Vite). See `.env.example` and `frontend/.env.example` for the full,
documented list. Key variables:

| Variable | Used by | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | backend | Signs/verifies auth tokens — must be a long random secret |
| `DATABASE_URL` | backend | PostgreSQL connection string |
| `CORS_ORIGINS` | backend | Comma-separated list of allowed frontend origins |
| `UPLOAD_DIR` / `MODELS_DIR` | backend | Local storage locations (swappable for S3/MinIO later) |
| `EYE_DATASET_DIR` | ml pipeline | Where to read the training dataset from |
| `EYE_EPOCHS` / `EYE_FINE_TUNE_EPOCHS` | ml pipeline | Deep-learning training length |
| `VITE_API_BASE_URL` | frontend | Where the frontend sends API requests |

Nothing is hard-coded: no credentials, secrets, file paths, or debug flags appear directly in
source.

## Running Locally

### 1. Start PostgreSQL
Either run Postgres yourself, or just start the `db` service from Compose:
```bash
docker compose up -d db
```

### 2. Run database migrations
```bash
cd backend
alembic upgrade head
```

### 3. Start the backend
```bash
# from backend/, with PYTHONPATH including the repo root so `ml` is importable
cd backend
PYTHONPATH=..:. uvicorn app.main:app --reload --port 8000
```
API docs are now at `http://localhost:8000/docs` (Swagger) and `/redoc` (ReDoc).

### 4. Start the frontend
```bash
cd frontend
npm run dev
```
App is now at `http://localhost:5173`.

> Until you've trained models (see below), `/api/predict` correctly returns **503** with a clear
> message instead of crashing — the frontend surfaces this as a toast.

## Running with Docker

```bash
docker compose up --build
```

This starts PostgreSQL, runs Alembic migrations, starts the backend on `:8000`, and serves the
built frontend (via Nginx, proxying `/api` to the backend) on `:8080`.

Trained model artifacts and uploaded images are bind-mounted (`./models`, `./uploads`,
`./ml/artifacts`) so they persist across container restarts and are easy to inspect.

## Training Models

Training runs **outside** the API container against your local dataset (training deep models is
resource-intensive; keeping it separate from the request/response path is intentional).

```bash
# from the repo root, with the ml/ dependencies installed
python -m ml.data.analyze_dataset        # dataset stats + distribution plots
python -m ml.training.train_all          # trains every traditional + deep model,
                                          # evaluates, compares, and selects the production model
```

Or train pieces individually:
```bash
python -m ml.training.train_traditional
python -m ml.training.train_deep --model MobileNet
python -m ml.evaluation.compare_models   # rebuild the comparison table + select best model
```

Artifacts land in:
- `models/` — one file per trained model, plus `production_model.keras` / `.joblib` and
  `production_model_metadata.json` (which the backend reads at startup)
- `ml/artifacts/reports/` — JSON metrics per model, `model_comparison.json`, `dataset_statistics.json`
- `ml/artifacts/plots/` — training curves, confusion matrices, comparison bar charts

After training, seed the `model_performance` table so the admin/API layer can also read metrics
from PostgreSQL (in addition to the JSON report the frontend already reads directly):
```bash
python scripts/seed_model_performance.py
```

Restart the backend (or redeploy) so it picks up the new `production_model_metadata.json` at
startup.

## Evaluating Models

```bash
python -m ml.evaluation.evaluate --model MobileNet
```
Re-prints the stored metrics for a single already-trained model without retraining.

## Model Comparison & Selection

Every model is evaluated on the **same untouched test split** using: accuracy, precision
(macro), recall/sensitivity (macro), specificity (macro, derived from the confusion matrix),
F1 (macro & weighted), ROC-AUC (macro, one-vs-rest), inference time (ms/image), training time,
parameter count, and model size.

**The production model is selected by a weighted composite score, not accuracy alone**
(see `ml/config.py::SELECTION_WEIGHTS` and `ml/evaluation/compare_models.py`):

```
score = 0.40 × macro_F1
      + 0.25 × balanced_accuracy
      + 0.20 × macro_recall (sensitivity)
      + 0.15 × macro_ROC_AUC
```

Accuracy alone is avoided because a model that always predicts the majority class can score well
on accuracy while failing the minority disease classes that matter most to catch. This logic, and
the resulting winner, are also surfaced in the frontend's Model Comparison page under "Model
Analysis."

## API Documentation

FastAPI auto-generates interactive docs at `/docs` (Swagger UI) and `/redoc`. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create an account, returns a JWT |
| POST | `/api/auth/login` | Log in, returns a JWT |
| GET | `/api/auth/me` | Current authenticated user |
| POST | `/api/predict` | Predict disease class for one image |
| POST | `/api/predict/batch` | Predict for multiple images |
| GET | `/api/models` | List available models + current production model |
| GET | `/api/model-performance` | Full model comparison report |
| GET | `/api/predictions` | Current user's prediction history (filterable) |
| GET | `/api/predictions/{id}` | Full detail for one prediction |
| GET | `/api/predictions/{id}/image` | Original uploaded image |
| GET | `/api/predictions/{id}/gradcam` | Grad-CAM overlay image |
| GET | `/api/dataset/statistics` | Dataset size / class balance / split stats |
| GET | `/api/admin/overview` | Admin-only platform statistics |
| GET | `/api/health` | Health check + model-loaded status |

Example prediction response:
```json
{
  "prediction_id": "5b1e...",
  "prediction": "Glaucoma",
  "confidence": 0.91,
  "probabilities": {
    "Cataract": 0.02,
    "Diabetic Retinopathy": 0.04,
    "Glaucoma": 0.91,
    "Normal": 0.03
  },
  "model": "EfficientNet",
  "model_version": "v1",
  "gradcam_available": true,
  "gradcam_url": "/api/predictions/5b1e.../gradcam",
  "disclaimer": "This prediction is generated by an AI/ML model for educational and research purposes..."
}
```

## Database Structure

- **users**: id, name, email (unique), password_hash, role (`user`/`admin`), created_at
- **predictions**: id, user_id (FK), image_path, predicted_class, confidence, model_name,
  model_version, gradcam_path, created_at
- **prediction_probabilities**: id, prediction_id (FK, unique), one probability column per class
- **model_performance**: id, model_name, model_version, accuracy, precision, recall, f1_score,
  specificity, roc_auc, inference_time_ms, training_time_seconds, n_parameters, model_size_kb,
  is_production, created_at

Schema is managed entirely through Alembic migrations (`backend/alembic/versions/`) — never
created by hand.

## Project Structure

```
eye-disease-prediction/
├── frontend/            React + TypeScript + Vite + Tailwind
│   └── src/{components,pages,layouts,hooks,services,context,types}
├── backend/              FastAPI app
│   └── app/{api,core,models,schemas,services,repositories,ml}
├── ml/                    Training pipeline (shared with backend)
│   └── {data,preprocessing,features,models,training,evaluation,explainability}
├── dataset/               Your images go here (cataract/diabetic_retinopathy/glaucoma/normal)
├── models/                Trained model artifacts + production_model_metadata.json
├── uploads/                User-uploaded images + Grad-CAM overlays
├── scripts/                 Utility scripts (e.g. seed_model_performance.py)
├── docker-compose.yml
├── .env.example
└── README.md
```

## Testing

**Backend** (pytest, SQLite in-memory DB, FastAPI TestClient):
```bash
cd backend
pytest
```
Covers registration/login, auth-protected routes, file-type/size validation, and the
predict-endpoint's graceful 503 behavior before any model has been trained.

**Frontend** (Vitest + Testing Library):
```bash
cd frontend
npm run test
```

**ML**: preprocessing output shape/range, feature-extraction vector shape, and model-loading are
exercised implicitly by the training scripts; extend `backend/tests/` or add `ml/tests/` for
deeper unit coverage as your dataset and models stabilize.

## Screenshots

_Add screenshots here after running the app against your dataset:_
- `docs/screenshots/landing.png`
- `docs/screenshots/predict.png`
- `docs/screenshots/model-comparison.png`
- `docs/screenshots/dataset-statistics.png`
- `docs/screenshots/admin-dashboard.png`

## Future Improvements

- Migrate local file storage to S3/MinIO using the existing storage-service abstraction
- Add cross-validation reporting for deep models, not just traditional ML
- Add per-class ROC curves and precision-recall curves as saved plots
- Add model versioning/rollback (multiple production candidates, A/B serving)
- Add rate limiting and request throttling on `/api/predict`
- Add CI (GitHub Actions) running backend/frontend test suites on every push

## Limitations

- Model performance is entirely dependent on the size, quality, and diversity of the dataset you
  provide. Small or imbalanced datasets will not generalize to real-world clinical populations.
- Grad-CAM highlights what the model attended to — it does not confirm clinical relevance.
- Traditional ML models use hand-engineered features (HOG/color/statistics), which may cap their
  achievable accuracy relative to the deep models regardless of dataset size.
- This project has not been validated against a certified medical benchmark and must not be used
  for real clinical decision-making.

## Medical Disclaimer

This system is an AI-assisted research and educational tool and is **not** a medical diagnosis.
Predictions may be inaccurate. Always consult a qualified eye-care professional for clinical
evaluation.

## Authors / Contributors

Generated as a complete academic reference implementation. 
Prathik R Sheregar 