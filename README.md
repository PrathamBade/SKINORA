# SKINORA

> **AI-powered Skin Health Assessment System**

SKINORA analyses skin images to assess characteristics such as acne severity. It is built as an academic software engineering project, combining a FastAPI backend with a PyTorch ML pipeline.

> ⚠️ **Disclaimer:** SKINORA is a prototype for academic purposes only. It is **not** a medical diagnosis tool. Always consult a licensed healthcare professional for skin-related medical advice.

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| **Phase 1** — Backend Foundation | ✅ Complete | FastAPI backend, auth, image upload, DB layer |
| **Phase 2** — ML/Data Pipeline | ✅ Complete | ACNE04 pipeline, ResNet18 trained (57.4% test acc) |
| **Phase 3** — Backend ↔ ML Integration | ✅ Complete | Live inference in API, status endpoint, full test suite |
| **Phase 4** — Frontend | 🔄 Next | React + Vite UI |
| **Phase 5** — Full Integration | ⏳ Planned | End-to-end |
| **Phase 6** — Testing & Deployment | ⏳ Planned | CI/CD, docs |

---

## Technology Stack

### Backend
- **Python 3.14** + **FastAPI** + **Uvicorn**
- **SQLAlchemy 2.x** (async) + **SQLite** (dev) → PostgreSQL (production)
- **Pydantic v2** for validation
- **JWT** (python-jose) + **bcrypt** (passlib) for auth

### Machine Learning
- **PyTorch** + **Torchvision**
- **OpenCV**, **NumPy**, **Pandas**, **Pillow**, **Scikit-learn**
- Dataset: **ACNE04** (`all_1024` — 1406 images, 4 severity classes)

### Frontend *(Phase 4)*
- React + JSX, Vite, JavaScript

---

## Project Structure

```
SKINORA/
├── .env.example              # Environment variable template
├── .gitignore
├── README.md
│
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py           # App entry point, router registration
│   │   ├── api/
│   │   │   ├── dependencies.py   # JWT auth dependency
│   │   │   └── routes/
│   │   │       ├── health.py
│   │   │       ├── auth.py
│   │   │       ├── users.py
│   │   │       ├── analysis.py
│   │   │       ├── observations.py
│   │   │       └── recommendations.py
│   │   ├── core/
│   │   │   ├── config.py     # Settings (pydantic-settings)
│   │   │   └── security.py   # JWT + bcrypt
│   │   ├── db/
│   │   │   └── database.py   # Engine, session, Base, init_db
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   └── services/         # Business logic
│   ├── tests/
│   │   ├── conftest.py       # Fixtures, isolated DB per test
│   │   ├── test_health.py
│   │   ├── test_auth.py
│   │   └── test_analysis.py
│   ├── uploads/              # Uploaded images (gitignored)
│   ├── requirements.txt
│   ├── pyproject.toml        # pytest config
│   └── README.md
│
├── frontend/                 # React app (Phase 4 — not built yet)
│
└── ml/
    ├── datasets/
    │   ├── raw/ACNE04/       # Raw dataset (gitignored)
    │   └── processed/        # CSV splits (gitignored)
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb
    │   └── 02_data_pipelin.ipynb
    ├── preprocessing/
    ├── training/
    ├── models/               # Saved model weights (gitignored)
    └── results/
```

---

## Backend Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone and navigate

```bash
git clone <repo-url>
cd SKINORA
```

### 2. Create virtual environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp ../.env.example .env
# Edit .env and set a real SECRET_KEY:
# python -c "import secrets; print(secrets.token_hex(32))"
```

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

Server runs at: `http://127.0.0.1:8000`

---

## API Documentation

Once running, visit:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Endpoints Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Welcome message |
| `GET` | `/health` | No | Health check |
| `POST` | `/api/v1/auth/register` | No | Create account |
| `POST` | `/api/v1/auth/login` | No | Get JWT token |
| `GET` | `/api/v1/users/me` | JWT | Current user profile |
| `POST` | `/api/v1/analysis/upload` | JWT | Upload skin image |
| `GET` | `/api/v1/analysis/history` | JWT | Past analyses (paginated) |
| `GET` | `/api/v1/analysis/{id}` | JWT | Single analysis |
| `GET` | `/api/v1/observations/{analysis_id}` | JWT | Analysis observations |
| `GET` | `/api/v1/recommendations` | JWT | Skincare guidance |

### Authentication

All protected endpoints require a Bearer token:

```
Authorization: Bearer <access_token>
```

---

## Environment Variables

See `.env.example` for all variables. Key ones:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(required)* | JWT signing key — generate with `secrets.token_hex(32)` |
| `DATABASE_URL` | `sqlite+aiosqlite:///./skinora.db` | SQLAlchemy async DB URL |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT expiry |
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS origins |
| `UPLOAD_DIR` | `uploads` | Image upload directory |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max upload size in MB |

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Expected: **26 passed**

Tests cover:
- System endpoints (`/`, `/health`, `/docs`, `/openapi.json`)
- Registration (success, duplicate email/username, weak password, invalid email)
- Login (success, wrong password, non-existent user)
- Protected route (`/me`) with valid, missing, and invalid tokens
- Image upload (JPEG, PNG, unsupported type, corrupt file, unauthenticated)
- Analysis status after upload (`awaiting_model`)
- Analysis history and retrieval
- Recommendations (no severity, all levels, invalid level)

---

## Dataset Setup

1. Download the **ACNE04** dataset
2. Place it at: `ml/datasets/raw/ACNE04/`
3. The `all_1024/` subfolder is used for the baseline model
4. Run `ml/notebooks/01_dataset_exploration.ipynb` to generate `acne_dataset.csv`
5. Run `ml/notebooks/02_data_pipelin.ipynb` to create train/val/test splits

---

## ML Status

| Component | Status | Detail |
|---|---|---|
| Dataset acquired (ACNE04 `all_1024`) | ✅ | 1406 images, 4 severity classes |
| Dataset exploration notebook | ✅ | `01_dataset_exploration.ipynb` |
| `build_dataset.py` — metadata → CSVs | ✅ | 984 train / 211 val / 211 test |
| `dataset.py` — PyTorch AcneDataset | ✅ | Augmented train transforms |
| `model.py` — ResNet18 builder | ✅ | Frozen backbone, 2,052 trainable params |
| `train.py` — Training loop | ✅ | Weighted CE loss, Adam, ReduceLROnPlateau |
| **Trained model** (`acne_resnet18.pth`) | ✅ | 15 epochs, CPU, 5.9 min |
| `evaluate.py` — Test-set evaluation | ✅ | See results below |
| `inference.py` — AcneInferenceService | ✅ | Ready for Phase 3 integration |
| ML → API integration | 🔄 Phase 3 | Next step |

### Baseline Model Results (Test Set — 211 images)

| Metric | Value |
|---|---|
| **Test Accuracy** | **57.35%** |
| Macro F1 | 0.5675 |
| Weighted F1 | 0.5780 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Level 0 (Clear) | 0.657 | 0.595 | 0.624 | 74 |
| Level 1 (Mild) | 0.600 | 0.548 | 0.573 | 93 |
| Level 2 (Moderate) | 0.359 | 0.519 | 0.424 | 27 |
| Level 3 (Severe) | 0.600 | 0.706 | 0.649 | 17 |

> **Context:** Random baseline = 25%. Frozen ResNet18 head-only training on CPU.
> Level 2 is weakest (only 177 samples). Phase 3 will explore backbone unfreezing and class-balanced sampling.

### How to reproduce

```bash
cd ml
python -m preprocessing.build_dataset   # Build CSVs
python -m training.train                # Train model (~6 min on CPU)
python -m training.evaluate             # Evaluate on test set
```

1. Branch from `main`
2. Follow PEP 8 and add type hints
3. Write tests for new endpoints
4. Run `pytest` before submitting