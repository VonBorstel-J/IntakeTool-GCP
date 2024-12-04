#config.py
import os
import logging
from secrets_utils import get_secret
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


def validate_environment_variables():
    """
    Validate that all required environment variables are set.
    """
    required_vars = ["GCP_PROJECT_ID", "DOCUMENT_AI_PROCESSOR_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


# Set GOOGLE_APPLICATION_CREDENTIALS for local development
if not os.getenv("GOOGLE_CLOUD_PROJECT"):  # Not running in GCP
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("email-parser", "service.json")


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or Google Cloud Secret Manager.
    """

    # Gmail API Configuration
    GMAIL_API_KEY: SecretStr = Field(..., env="GMAIL_API_KEY")

    # Gemini API Configuration
    GEMINI_API_KEY: SecretStr = Field(..., env="GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = Field(..., env="GEMINI_MODEL_NAME")
    GEMINI_ENDPOINT: str = Field(..., env="GEMINI_ENDPOINT")
    ATTACHMENT_OCR_PROCESSOR_ID: str = Field(..., env="ATTACHMENT_OCR_PROCESSOR_ID")
    ATTACHMENT_OCR_REGION: str = Field("us", env="ATTACHMENT_OCR_REGION")

    # GCP Configuration
    GCP_PROJECT_ID: str = Field(..., env="GCP_PROJECT_ID")
    GCP_LOCATION: str = Field("us-central1", env="GCP_LOCATION")

    # BigQuery Configuration
    BIGQUERY_PROJECT_ID: str = Field(..., env="BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET: str = Field(..., env="BIGQUERY_DATASET")

    # Logging Configuration
    LOG_FILE: str = Field("app.log", env="LOG_FILE")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # Additional Settings
    EMAIL_ATTACHMENT_BUCKET: str = Field(..., env="EMAIL_ATTACHMENT_BUCKET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Load settings
settings = Settings()

# Validate environment variables
validate_environment_variables()

# Load secrets if running in GCP
if os.getenv("GCP_PROJECT_ID"):
    # Overwrite settings with secrets from Google Secret Manager
    settings.GMAIL_API_KEY = get_secret("GMAIL_API_KEY")
    settings.GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_secret("GOOGLE_APPLICATION_CREDENTIALS")

# Configure logging
LOG_LEVEL = settings.LOG_LEVEL.upper()
LOG_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

def get_logger(extra: dict = None) -> logging.LoggerAdapter:
    logger = logging.getLogger("app_logger")
    if extra is None:
        extra = {}
    return logging.LoggerAdapter(logger, extra)
