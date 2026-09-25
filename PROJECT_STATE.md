# SKINORA — Project State, Architecture & Agent Runbook

> **Audience:** This document is designed for AI coding assistants and software engineers picking up this project on any development environment or machine. It details the complete system state, architectural decisions, completed phases, execution runbook, and deployment guidelines.

---

## 1. Executive Summary

**SKINORA** is an AI-powered skin health assessment and personalized skincare assistance platform.  
The core workflow is:
1. User authenticates via token-based JWT authentication (`/api/v1/auth/*`).
2. User uploads a skin image (`/api/v1/analysis/upload`).
3. The backend validates file format, image integrity (Pillow), and file size.
4. An integrated deep learning model (**PyTorch ResNet18**) analyzes the skin image and predicts acne severity (Levels 0–3) with softmax confidence scores.
5. Structured observations are persisted in an asynchronous relational database.
6. The system provides evidence-based, non-medical skincare routine recommendations (`/api/v1/recommendations`) based on detected severity.
7. An upcoming React + Vite frontend (Phase 4) will consume these APIs.

---

## 2. Project Status & Phase Roadmap

| Phase | Milestone | Status | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Backend Foundation** | ✅ **Complete** | FastAPI app, async SQLite engine, Pydantic v2 schemas, JWT auth, secure image upload, 26 unit tests. |
| **Phase 2** | **ML & Data Pipeline** | ✅ **Complete** | ACNE04 dataset preprocessing, stratified splits, ResNet18 baseline trained (57.35% test accuracy), evaluation metrics. |
| **Phase 3** | **Backend ↔ ML Integration**| ✅ **Complete** | Auto-initializing `BackendInferenceService` singleton, live prediction in `/api/v1/analysis/upload`, 28/28 tests passing. |
| **Phase 4** | **Frontend Application** | 🔄 **Next** | React + JSX + Vite SPA (login, register, image upload, dashboard, routine guidance). *Do not use TypeScript.* |
| **Phase 5** | **Full Integration** | ⏳ **Planned** | Frontend-backend end-to-end integration, error boundary handling, user session management. |
| **Phase 6** | **Testing & Deployment** | ⏳ **Planned** | Containerization, CI/CD pipeline, production PostgreSQL deployment, reverse proxy setup. |

---

## 3. Technology Stack

### Backend
- **Language:** Python 3.10+ (tested on Python 3.14.6)
- **Framework:** FastAPI `0.141.1`, Uvicorn `0.53.0`, Starlette `1.7.0`
- **Validation:** Pydantic `2.13.5`, Pydantic-Settings `2.15.0`, Email-Validator `2.3.0`
- **Database & ORM:** SQLAlchemy `2.1.0` (async), `aiosqlite 0.22.1`, `greenlet 3.5.6`
- **Authentication & Security:** `python-jose 3.5.0` (JWT HS256), `passlib 1.7.4` + `bcrypt 4.0.1`
- **Image Validation:** Pillow `12.3.0`
- **Testing:** Pytest `9.1.1`, Pytest-Asyncio `1.4.0`, HTTPX `0.28.1`

### Machine Learning
- **Framework:** PyTorch `2.14.0+cpu`, Torchvision `0.29.0+cpu`
- **Data Processing:** Pandas `3.0.6`, NumPy `2.5.3`, Scikit-learn `1.9.1`
- **Dataset:** ACNE04 (`all_1024` — 1,406 total images, 1024×1024 RGB JPGs)
- **Model Checkpoint:** `ml/models/acne_resnet18.pth` (44.8 MB)

### Upcoming Frontend (Phase 4)
- React + JSX
- Vite
- JavaScript (*No TypeScript unless explicitly requested*)
- Tailwind CSS

---

## 4. Repository Structure

```
SKINORA/
├── .env.example                     # Environment template (copy to backend/.env)
├── .gitignore                       # Git ignore policy (protects datasets, .env, DB)
├── README.md                        # Primary human-readable README
├── PROJECT_STATE.md                 # THIS FILE — comprehensive agent & system runbook
│
├── backend/                         # FastAPI Application Root
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # Entrypoint, CORS, lifespan, global error handlers
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py      # get_current_user JWT bearer dependency
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py          # /api/v1/auth/register, /api/v1/auth/login
│   │   │       ├── users.py         # /api/v1/users/me
│   │   │       ├── analysis.py      # /api/v1/analysis/upload, /history, /{id}
│   │   │       ├── observations.py  # /api/v1/observations/{analysis_id}
│   │   │       ├── recommendations.py # /api/v1/recommendations
│   │   │       ├── ml.py            # /api/v1/ml/status
│   │   │       └── health.py        # /health
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py            # Settings with robust model path resolver
│   │   │   └── security.py          # hash_password, verify_password, JWT tokens
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── database.py          # Async engine, sessionmaker, get_db, init_db
│   │   │
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   └── inference_wrapper.py # BackendInferenceService singleton bridge
│   │   │
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── user.py              # User table (UUID, email, hashed_password)
│   │   │   ├── analysis.py          # Analysis table (status, image_path, FK to user)
│   │   │   └── observation.py       # Observation table (type, value, confidence)
│   │   │
│   │   ├── schemas/                 # Pydantic validation schemas
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # RegisterRequest, LoginRequest, TokenResponse
│   │   │   ├── user.py              # UserResponse (never leaks hashed_password)
│   │   │   ├── analysis.py          # UploadResponse, AnalysisResponse, MLStatus
│   │   │   └── observation.py       # ObservationResponse
│   │   │
│   │   └── services/                # Business logic layer
│   │       ├── __init__.py
│   │       ├── auth_service.py      # User creation & credential verification
│   │       ├── analysis_service.py  # Image validation, storage, inference dispatch
│   │       └── recommendation_service.py # Evidence-based skincare guidelines
│   │
│   ├── tests/                       # Automated Pytest Suite (28 tests)
│   │   ├── __init__.py
│   │   ├── conftest.py              # Test client, in-memory DB isolation, mock images
│   │   ├── test_health.py           # Root, health, and docs accessibility
│   │   ├── test_auth.py             # Register, duplicate handling, login, /me auth
│   │   └── test_analysis.py         # Upload validation, ML prediction, observations
│   │
│   ├── uploads/                     # Upload directory (.gitkeep tracked; images ignored)
│   ├── pyproject.toml               # Pytest configuration
│   └── requirements.txt             # Pinned backend dependencies
│
└── ml/                              # Machine Learning Pipeline
    ├── __init__.py
    ├── inference.py                 # Standalone AcneInferenceService class
    │
    ├── datasets/
    │   ├── raw/ACNE04/              # Gitignored raw dataset (acne_1024/all_1024/)
    │   └── processed/               # Gitignored CSV splits (acne_dataset, train, val, test)
    │
    ├── models/
    │   ├── .gitkeep
    │   └── acne_resnet18.pth        # Trained weights (44.8 MB, committed in repo)
    │
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb
    │   └── 02_data_pipelin.ipynb
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── build_dataset.py         # metadata.jsonl parser -> CSV splits
    │   └── dataset.py               # PyTorch AcneDataset & DataLoaders with augmentation
    │
    ├── results/                     # Training artifacts & plots
    │   ├── confusion_matrix.png
    │   ├── training_curves.png
    │   ├── evaluation_results.json
    │   └── training_history.json
    │
    └── training/
        ├── __init__.py
        ├── config.py                # Hyperparameters, paths, class names
        ├── model.py                 # ResNet18 architecture builder
        ├── train.py                 # Weighted cross-entropy training loop
        └── evaluate.py              # Test set evaluation & confusion matrix generation
```

---

## 5. Machine Learning Pipeline Details

### Baseline Model: ResNet18
- **Input:** 224×224 RGB image normalized using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Architecture:** Pretrained ResNet18 with a frozen backbone (all layers frozen except final classifier head `fc`).
- **Trainable Parameters:** 2,052 (out of 11,178,564 total).
- **Optimization:** Adam (`lr=1e-3`, `weight_decay=1e-4`), `ReduceLROnPlateau` scheduler.
- **Loss:** CrossEntropyLoss with inverse class frequency weights to counteract class imbalance:
  - Class 0 (Clear): 491 images
  - Class 1 (Mild): 623 images
  - Class 2 (Moderate): 177 images
  - Class 3 (Severe): 115 images
- **Split Ratio:** 70% Train (984) / 15% Val (211) / 15% Test (211), stratified by acne severity.

### Performance on Held-Out Test Set (211 images)
- **Test Accuracy:** **57.35%** (random baseline = 25.0%)
- **Macro F1:** **0.5675**
- **Weighted F1:** **0.5780**
- **Per-Class Metrics:**
  - `Level 0 (Clear)`: Precision 65.7%, Recall 59.5%, F1 **0.624** (74 samples)
  - `Level 1 (Mild)`: Precision 60.0%, Recall 54.8%, F1 **0.573** (93 samples)
  - `Level 2 (Moderate)`: Precision 35.9%, Recall 51.9%, F1 **0.424** (27 samples)
  - `Level 3 (Severe)`: Precision 60.0%, Recall 70.6%, F1 **0.649** (17 samples)

### ML Artifacts
- Checkpoint: `ml/models/acne_resnet18.pth`
- Visual Curves: `ml/results/training_curves.png`
- Confusion Matrix: `ml/results/confusion_matrix.png`
- Metrics Data: `ml/results/evaluation_results.json`

---

## 6. Backend API Specification

All application endpoints are prefixed with `/api/v1/`.

| Method | Endpoint | Auth | Request Body | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | None | None | Welcome message, API version, and docs link. |
| `GET` | `/health` | None | None | Returns `{"status": "healthy", "service": "SKINORA API"}`. |
| `GET` | `/api/v1/ml/status` | None | None | Returns `{model_loaded: bool, model_path: str, error: null}`. |
| `POST` | `/api/v1/auth/register` | None | JSON: `{email, username, password, full_name}` | Creates user; returns JWT token. Status 201. |
| `POST` | `/api/v1/auth/login` | None | JSON: `{email, password}` | Verifies credentials; returns JWT token. |
| `GET` | `/api/v1/users/me` | Bearer JWT | None | Returns authenticated user profile (no password hash). |
| `POST` | `/api/v1/analysis/upload` | Bearer JWT | `multipart/form-data`: `file` (image) | Validates image, runs ResNet18 inference, saves DB record. |
| `GET` | `/api/v1/analysis/history` | Bearer JWT | Query: `limit`, `offset` | Paginated list of past analyses for current user. |
| `GET` | `/api/v1/analysis/{id}` | Bearer JWT | None | Detailed analysis record including status & observations. |
| `GET` | `/api/v1/observations/{analysis_id}` | Bearer JWT | None | Returns skin observations (severity tag, confidence, description). |
| `GET` | `/api/v1/recommendations` | Bearer JWT | Query: `acne_severity` (0–3, optional) | Routine guidance tailored to severity level. |

---

## 7. Local Developer Runbook

### Step 1: Environment Setup
Clone the repository and enter the directory:
```bash
git clone https://github.com/PrathamBade/SKINORA.git
cd SKINORA
```

### Step 2: Virtual Environment & Dependencies
Create and activate a virtual environment:
```bash
# Windows
python -m venv backend/venv
backend\venv\Scripts\activate

# macOS / Linux
python3 -m venv backend/venv
source backend/venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `backend/.env`:
```bash
cp .env.example backend/.env
```
*(On Windows PowerShell: `Copy-Item .env.example backend/.env`)*

In `backend/.env`, set your `SECRET_KEY`:
```ini
APP_NAME=SKINORA API
APP_VERSION=1.0.0
DEBUG=true

SECRET_KEY=generate-a-32-byte-hex-secret
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

DATABASE_URL=sqlite+aiosqlite:///./skinora.db
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE_MB=10

ML_MODEL_PATH=../ml/models/acne_resnet18.pth
```

### Step 4: Run the Automated Test Suite
From the `backend/` directory:
```bash
cd backend
python -m pytest tests/ -v
```
**Expected outcome:** All 28 tests pass with 0 failures and 0 warnings.

### Step 5: Start the FastAPI Server
From the `backend/` directory:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open in browser:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Probe: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- ML Model Status: [http://127.0.0.1:8000/api/v1/ml/status](http://127.0.0.1:8000/api/v1/ml/status)

---

## 8. Reproducing the Machine Learning Pipeline

If you wish to re-preprocess the data or retrain the ResNet18 model:

1. **Obtain ACNE04:** Ensure the ACNE04 dataset is placed at `ml/datasets/raw/ACNE04/acne_1024/all_1024/`.
2. **Build Dataset Splits:**
   ```bash
   cd ml
   python -m preprocessing.build_dataset
   ```
   This generates `ml/datasets/processed/` containing `acne_dataset.csv`, `train.csv`, `val.csv`, and `test.csv`.
3. **Train Model:**
   ```bash
   python -m training.train
   ```
   Trains ResNet18 for 15 epochs on CPU (~6 minutes) and saves the best model to `ml/models/acne_resnet18.pth`.
4. **Evaluate Model:**
   ```bash
   python -m training.evaluate
   ```
   Evaluates on the 211 held-out test images and outputs `ml/results/evaluation_results.json` and `confusion_matrix.png`.

---

## 9. Production Deployment Guide

### A. Environment Configuration
In production environments (e.g. AWS EC2, GCP Compute, Render, Railway, DigitalOcean):
- Generate a cryptographically strong secret: `python -c "import secrets; print(secrets.token_hex(32))"`
- Set `DEBUG=false`.
- Configure `ALLOWED_ORIGINS` to match your deployed frontend domain.

### B. PostgreSQL Database Setup
To switch from SQLite to PostgreSQL:
1. Install asyncpg: `pip install asyncpg`
2. Update `DATABASE_URL` in `.env`:
   ```ini
   DATABASE_URL=postgresql+asyncpg://<username>:<password>@<db-host>:5432/<dbname>
   ```
3. No code changes are required in `app/db/database.py` because SQLAlchemy's async engine abstracts dialect differences.

### C. Production ASGI Execution (Gunicorn + Uvicorn Workers)
Run behind a process manager:
```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### D. Reverse Proxy (Nginx)
Configure Nginx with a 10MB client max body size to allow image uploads:
```nginx
server {
    listen 80;
    server_name api.skinora.yourdomain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### E. Dockerfile Reference
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application and trained ML model
COPY backend/app ./app
COPY ml/models/acne_resnet18.pth /app/ml/models/acne_resnet18.pth
COPY ml /app/ml

ENV PYTHONPATH=/app
ENV ML_MODEL_PATH=/app/ml/models/acne_resnet18.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10. Critical Engineering Decisions & Gotchas

1. **Model Path Resolution:**
   The property `resolved_model_path` in `backend/app/core/config.py` searches both current working directory and relative repository root (`SKINORA/ml/models/`). This prevents model load failures regardless of whether `uvicorn` or `pytest` is invoked from repository root or `backend/`.

2. **On-Demand Model Initialization:**
   `BackendInferenceService` in `backend/app/ml/inference_wrapper.py` auto-initializes upon first request if the FastAPI lifespan was bypassed (e.g. during script execution or test harnesses).

3. **Passlib & Bcrypt Compatibility:**
   `passlib` has a known incompatibility with `bcrypt >= 5.0.0` (missing `__about__`). Always pin `bcrypt==4.0.1`.

4. **Async SQLAlchemy & Greenlet:**
   Async SQLite (`aiosqlite`) requires `greenlet`. Both `aiosqlite` and `greenlet` are pinned in `requirements.txt`.

5. **Security & Git Hygiene:**
   - `.env` must never be committed.
   - Raw dataset images (`ml/datasets/`) must never be committed.
   - User uploaded images (`backend/uploads/*.jpg`) are gitignored via `.gitkeep`.
   - The trained model binary `acne_resnet18.pth` is tracked in Git (44.8 MB, comfortably under GitHub's 100 MB hard limit).

---

## 11. Instructions for Next Agent (Phase 4: Frontend)

When starting Phase 4 (Frontend):
1. **Directory:** Create a `frontend/` directory at the project root using Vite + React.
2. **Language:** Use **React + JSX + JavaScript**. *Do NOT use TypeScript* unless explicitly instructed by the user.
3. **Styling:** Use Tailwind CSS.
4. **Backend Integration:** The frontend must communicate directly with the FastAPI endpoints documented in Section 6.
5. **Key Views to Build:**
   - **Authentication:** Login (`/api/v1/auth/login`) and Register (`/api/v1/auth/register`) pages storing the JWT bearer token.
   - **Dashboard:** Welcome view showing user profile (`/api/v1/users/me`) and recent analyses.
   - **Image Upload & Analysis:** Drag-and-drop file uploader submitting to `/api/v1/analysis/upload`, displaying real-time acne severity results and confidence meters.
   - **Routine Recommendations:** Display evidence-based routine advice based on severity level fetched from `/api/v1/recommendations`.
   - **Analysis History:** Timeline view of past assessments fetched from `/api/v1/analysis/history`.
