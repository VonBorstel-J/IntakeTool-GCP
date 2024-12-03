#config.py
import os
from pydantic import BaseSettings, Field
from google.cloud import secretmanager
from config import get_secret


def get_secret(secret_id: str):
    """
    Fetch a secret from Google Cloud Secret Manager.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{os.getenv('GCP_PROJECT_ID')}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        raise Exception(f"Failed to fetch secret {secret_id}: {e}")

# Set GOOGLE_APPLICATION_CREDENTIALS only for local environments
if not os.getenv("GOOGLE_CLOUD_PROJECT"):  # Not running in GCP
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "email-parser\\service.json"
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_secret("google-application-credentials")


GEMINI_API_KEY = get_secret("gemini-api-key")
GMAIL_API_KEY = get_secret("gmail-api-key")
GOOGLE_SERVICE_ACCOUNT_PATH = get_secret("GOOGLE_SERVICE_ACCOUNT_PATH")
GCP_PROJECT_ID = get_secret("GCP_PROJECT_ID")
GCP_LOCATION = get_secret("GCP_LOCATION")
GEMINI_MODEL_NAME = get_secret("GEMINI_MODEL_NAME")
GEMINI_ENDPOINT = get_secret("GEMINI_ENDPOINT")
BIGQUERY_PROJECT_ID = get_secret("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET = get_secret("BIGQUERY_DATASET")
LOG_FILE = get_secret("LOG_FILE")
ATTACHMENT_OCR_PROCESSOR_ID = get_secret("ATTACHMENT_OCR_PROCESSOR_ID")
ATTACHMENT_OCR_REGION = get_secret("ATTACHMENT_OCR_REGION")
EMAIL_ATTACHMENT_BUCKET = get_secret("EMAIL_ATTACHMENT_BUCKET")

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or Google Cloud Secret Manager.
    """

    # Gmail API Configuration
    GMAIL_API_KEY: str = Field(..., env="GMAIL_API_KEY")
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = Field(..., env="GEMINI_MODEL_NAME")
    GEMINI_ENDPOINT: str = Field(..., env="GEMINI_ENDPOINT")
    ATTACHMENT_OCR_PROCESSOR_ID: str = Field(..., env="ATTACHMENT_OCR_PROCESSOR_ID")
    ATTACHMENT_OCR_REGION: str = Field("us", env="ATTACHMENT_OCR_REGION")
    
    # GCP Configuration
    GCP_PROJECT_ID: str = Field(..., env="GCP_PROJECT_ID")
    GCP_LOCATION: str = Field("us-central1", env="GCP_LOCATION")  # Placeholder default
    
    # BigQuery Configuration
    BIGQUERY_PROJECT_ID: str = Field(..., env="BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET: str = Field(..., env="BIGQUERY_DATASET")
    
    # Logging Configuration
    LOG_FILE: str = Field("app.log", env="LOG_FILE")
    
    # Additional Settings
    SOME_SETTING: str = Field("your_setting_value", env="SOME_SETTING")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()

# Fetch secrets from Google Secret Manager (override if running in cloud)
if os.getenv("GCP_PROJECT_ID"):
    os.environ["GEMINI_API_KEY"] = get_secret("GEMINI_API_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_secret("GOOGLE_APPLICATION_CREDENTIALS")
