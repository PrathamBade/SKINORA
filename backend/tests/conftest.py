"""
SKINORA Backend — pytest configuration and shared fixtures.

Phase 3 update: the inference service is initialized with the real trained
model path so upload tests can verify actual ML predictions are returned.
Each test function gets a fresh in-memory SQLite database for isolation.
"""

import io
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Override settings BEFORE importing app modules
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["UPLOAD_DIR"] = "test_uploads"

# Point at the real trained model — conftest lives in backend/tests/
# so we go: backend/tests/ -> backend/ -> SKINORA/ -> ml/models/
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "models" / "acne_resnet18.pth"
os.environ["ML_MODEL_PATH"] = str(_MODEL_PATH)

# Clear lru_cache so test env vars are picked up
from app.core.config import get_settings  # noqa: E402
get_settings.cache_clear()

from app.main import app  # noqa: E402
from app.db import database as db_module  # noqa: E402
from app.db.database import Base  # noqa: E402
from app.ml.inference_wrapper import BackendInferenceService  # noqa: E402


# ---------------------------------------------------------------------------
# Initialize inference service once for the test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def initialize_ml():
    """Load the trained model once for the whole test session."""
    BackendInferenceService.initialize(_MODEL_PATH)
    yield


# ---------------------------------------------------------------------------
# Per-test isolated in-memory database
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def isolated_db():
    """Create a fresh in-memory SQLite DB for each test."""
    from app.models import user, analysis, observation  # noqa: F401

    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, future=True,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(
        bind=test_engine, class_=AsyncSession,
        expire_on_commit=False, autoflush=False, autocommit=False,
    )

    original_engine = db_module.engine
    original_session = db_module.AsyncSessionLocal
    db_module.engine = test_engine
    db_module.AsyncSessionLocal = TestSessionLocal

    yield

    db_module.engine = original_engine
    db_module.AsyncSessionLocal = original_session
    await test_engine.dispose()


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(200, 150, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(180, 140, 110)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# User fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def registered_user(client: AsyncClient) -> dict:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@skinora.dev",
            "username": "testuser",
            "password": "securepassword123",
            "full_name": "Test User",
        },
    )
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    return resp.json()


@pytest_asyncio.fixture
async def auth_headers(registered_user: dict) -> dict:
    token = registered_user["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
