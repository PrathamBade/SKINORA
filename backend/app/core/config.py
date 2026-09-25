"""
SKINORA Backend — Core Configuration

Loads application settings from environment variables / .env file.
All secrets must be set via environment variables in production.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "SKINORA API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ------------------------------------------------------------------
    # Security / JWT
    # ------------------------------------------------------------------
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./skinora.db"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # ------------------------------------------------------------------
    # File Upload
    # ------------------------------------------------------------------
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # ML Inference
    # ------------------------------------------------------------------
    # Path to the trained model weights (.pth file).
    # Relative paths are resolved from the backend/ working directory.
    ml_model_path: str = "../ml/models/acne_resnet18.pth"

    @property
    def resolved_model_path(self) -> Path:
        """Return the absolute path to the ML model file, resolving across execution directories."""
        p = Path(self.ml_model_path)
        if p.is_absolute() and p.exists():
            return p

        # 1. Try relative to cwd
        candidate_cwd = (Path.cwd() / p).resolve()
        if candidate_cwd.exists():
            return candidate_cwd

        # 2. Try relative to SKINORA project root (config is in backend/app/core/config.py)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        candidate_root_models = (project_root / "ml" / "models" / p.name).resolve()
        if candidate_root_models.exists():
            return candidate_root_models

        candidate_rel_root = (project_root / p).resolve()
        if candidate_rel_root.exists():
            return candidate_rel_root

        return candidate_cwd


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.  Use as a FastAPI dependency."""
    return Settings()
