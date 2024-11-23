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

GEMINI_API_KEY = get_secret("gemini-api-key")
GMAIL_API_KEY = get_secret("gmail-api-key")

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
