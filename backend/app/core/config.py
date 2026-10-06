"""
All configuration comes from environment variables (see .env.example).
Nothing sensitive is hard-coded here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "Eye Disease Prediction API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api"

    # --- Security ---
    JWT_SECRET_KEY: str = "CHANGE_ME_IN_ENV"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h

    # --- Database ---
    DATABASE_URL: str = "postgresql://eye_user:eye_password@localhost:5432/eye_disease_db"

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # --- File storage ---
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/bmp,image/tiff"

    # --- ML ---
    MODELS_DIR: str = "/app/models"
    ARTIFACTS_DIR: str = "ml/artifacts"
    DEFAULT_MODEL_NAME: str = "auto"  # 'auto' = use production_model_metadata.json

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_image_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(",") if t.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_PATH = (BASE_DIR / settings.UPLOAD_DIR).resolve()
MODELS_PATH = (BASE_DIR / settings.MODELS_DIR).resolve()
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
