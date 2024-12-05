#config.py

import os
import logging
from secrets_utils import get_secret  # Ensure this utility is correctly implemented
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from typing import Optional, List


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or Google Cloud Secret Manager.
    """

    # Environment Configuration
    ENVIRONMENT: str = Field(default="local", env="ENVIRONMENT")  # "local" or "gcp"

    # Gmail API Configuration
    GMAIL_API_KEY: Optional[SecretStr] = Field(default=None, env="GMAIL_API_KEY")

    # Gemini API Configuration
    GEMINI_API_KEY: Optional[SecretStr] = Field(default=None, env="GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = Field(default="text-bison-001", env="GEMINI_MODEL_NAME")
    GEMINI_ENDPOINT: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:predict",
        env="GEMINI_ENDPOINT",
    )

    # GCP Configuration
    GCP_PROJECT_ID: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    GCP_LOCATION: str = Field(default="us-central1", env="GCP_LOCATION")

    # BigQuery Configuration
    BIGQUERY_PROJECT_ID: Optional[str] = Field(default=None, env="BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET: Optional[str] = Field(default=None, env="BIGQUERY_DATASET")

    # OAuth2 Configuration
    GOOGLE_CLIENT_SECRETS_FILE: Optional[str] = Field(
        default=None, env="GOOGLE_CLIENT_SECRETS_FILE"
    )
    GOOGLE_SCOPES: List[str] = Field(
        default=["https://www.googleapis.com/auth/gmail.readonly"], env="GOOGLE_SCOPES"
    )
    REDIRECT_URI: Optional[str] = Field(default=None, env="REDIRECT_URI")  # OAuth2 Redirect URI

    # Security Configuration
    SECRET_KEY: Optional[str] = Field(default=None, env="SECRET_KEY")  # For JWT signing

    # Allowed Hosts
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")  # Adjust as needed

    # Logging Configuration
    LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # Additional Settings
    EMAIL_ATTACHMENT_BUCKET: Optional[str] = Field(default=None, env="EMAIL_ATTACHMENT_BUCKET")

    # GCS Paths
    ATTACHMENTS_PATH: str = Field(default="attachments/", env="ATTACHMENTS_PATH")
    BIGQUERY_STAGING_PATH: str = Field(default="bigquery/", env="BIGQUERY_STAGING_PATH")
    LOGS_PATH: str = Field(default="logs/", env="LOGS_PATH")
    TEMP_PATH: str = Field(default="temp/", env="TEMP_PATH")

    # Document AI OCR
    ATTACHMENT_OCR_PROCESSOR_ID: Optional[str] = Field(default=None, env="ATTACHMENT_OCR_PROCESSOR_ID")
    ATTACHMENT_OCR_REGION: str = Field(default="us-central1", env="ATTACHMENT_OCR_REGION")
    DOCUMENT_AI_PROCESSOR_ID: Optional[str] = Field(default=None, env="DOCUMENT_AI_PROCESSOR_ID")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    # Maximum Concurrent Tasks
    MAX_CONCURRENT_TASKS: int = Field(default=5, env="MAX_CONCURRENT_TASKS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def validate_environment_variables(settings: Settings):
    """
    Validate that all required environment variables are set based on the environment.
    Log warnings for missing variables instead of raising an error.
    """
    required_vars = [
        "GOOGLE_CLIENT_SECRETS_FILE",
        "DOCUMENT_AI_PROCESSOR_ID",
        "ATTACHMENT_OCR_PROCESSOR_ID",
    ]

    # Identify missing variables
    missing_vars = [var for var in required_vars if not getattr(settings, var, None)]

    if missing_vars:
        # Log warnings for missing variables
        for var in missing_vars:
            print(f"Warning: Missing environment variable: {var}")

def load_secrets(settings: Settings):
    """
    Load secrets from Google Secret Manager if running in GCP.
    Log warnings for missing secrets instead of failing.
    """
    if settings.ENVIRONMENT == "gcp":
        try:
            # Attempt to fetch secrets
            settings.GMAIL_API_KEY = get_secret("GMAIL_API_KEY")
            settings.GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
            settings.SECRET_KEY = get_secret("SECRET_KEY")
            settings.DOCUMENT_AI_PROCESSOR_ID = get_secret("DOCUMENT_AI_PROCESSOR_ID")
            settings.ATTACHMENT_OCR_PROCESSOR_ID = get_secret("ATTACHMENT_OCR_PROCESSOR_ID")

            # Handle GOOGLE_APPLICATION_CREDENTIALS separately
            google_creds = get_secret("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if google_creds:
                with open("/tmp/service_account.json", "w") as f:
                    f.write(google_creds)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/service_account.json"

        except Exception as e:
            # Log a warning for missing secrets but allow the app to proceed
            print(f"Warning: Failed to load one or more secrets: {e}")


# Load settings
settings = Settings()

# Validate environment variables
validate_environment_variables(settings)

# Load secrets if running in GCP
load_secrets(settings)

# Configure logging
LOG_LEVEL = settings.LOG_LEVEL.upper()
LOG_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

log_dir = os.path.dirname(settings.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=settings.LOG_FILE, filemode='a')


def get_logger(extra: dict = None) -> logging.LoggerAdapter:
    logger = logging.getLogger("app_logger")
    if extra is None:
        extra = {}
    return logging.LoggerAdapter(logger, extra)


if settings.ENVIRONMENT == "local" and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("email-parser", "service.json")
