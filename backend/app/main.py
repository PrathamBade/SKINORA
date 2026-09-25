"""
SKINORA Backend — Application Entry Point

Initialises the FastAPI application, registers middleware, mounts
all API routers, and configures logging and exception handlers.

Start the server:
    uvicorn app.main:app --reload
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, auth, health, ml, observations, recommendations, users
from app.core.config import get_settings
from app.db.database import init_db

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

settings = get_settings()

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown events
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run initialisation tasks on startup and cleanup on shutdown."""
    logger.info("SKINORA API starting up — version %s", settings.app_version)

    # Initialise database
    await init_db()
    logger.info("Database tables initialised.")
    logger.info("Upload directory: %s", settings.upload_path.resolve())

    # Load ML inference model (non-fatal if not found)
    from app.ml.inference_wrapper import BackendInferenceService
    model_path = settings.resolved_model_path
    logger.info("Loading ML model from: %s", model_path)
    BackendInferenceService.initialize(model_path)
    if BackendInferenceService.is_available():
        logger.info("ML inference service: READY")
    else:
        logger.warning("ML inference service: NOT AVAILABLE — analyses will have status=awaiting_model")

    yield
    logger.info("SKINORA API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-powered skin health assessment API.\n\n"
        "**Phase 1 — Backend Foundation** is complete.\n"
        "ML inference integration will follow in Phase 3.\n\n"
        "> ⚠️ This is an academic prototype. "
        "It is NOT a medical diagnosis tool."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — never expose internal exception details to clients."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
            },
        },
    )


# ---------------------------------------------------------------------------
# Root endpoint (keep existing behaviour)
# ---------------------------------------------------------------------------


@app.get("/", tags=["System"], summary="Welcome")
async def root() -> dict[str, Any]:
    """Welcome message and API entry point."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# API v1 routers
# ---------------------------------------------------------------------------

API_V1_PREFIX = "/api/v1"

app.include_router(health.router)                                 # GET /health
app.include_router(auth.router, prefix=API_V1_PREFIX)            # /api/v1/auth/*
app.include_router(users.router, prefix=API_V1_PREFIX)           # /api/v1/users/*
app.include_router(analysis.router, prefix=API_V1_PREFIX)        # /api/v1/analysis/*
app.include_router(observations.router, prefix=API_V1_PREFIX)    # /api/v1/observations/*
app.include_router(recommendations.router, prefix=API_V1_PREFIX) # /api/v1/recommendations
app.include_router(ml.router, prefix=API_V1_PREFIX)              # /api/v1/ml/*