#config.py
import os
import logging
from pydantic import BaseSettings, Field, SecretStr
from secrets import get_secret
from google.cloud import secretmanager

def validate_environment_variables():
    required_vars = ["GCP_PROJECT_ID", "GCS_BUCKET_NAME", "DOCUMENT_AI_PROCESSOR_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set GOOGLE_APPLICATION_CREDENTIALS only for local environments
if not os.getenv("GOOGLE_CLOUD_PROJECT"):  # Not running in GCP
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "email-parser\\service.json"
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_secret("google-application-credentials")

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

# Fetch secrets from Google Secret Manager (override if running in cloud)
if os.getenv("GCP_PROJECT_ID"):
    os.environ["GEMINI_API_KEY"] = get_secret("GEMINI_API_KEY")
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
