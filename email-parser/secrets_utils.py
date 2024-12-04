# secrets_utils.py
import os
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound, PermissionDenied
from config import settings, get_logger

logger = get_logger({"module": "secrets_utils"})

def get_secret(secret_id: str) -> Optional[str]:
    """
    Fetch a secret from Google Cloud Secret Manager.

    Args:
        secret_id (str): The ID of the secret to fetch.

    Returns:
        Optional[str]: The secret value if fetched successfully; otherwise, None.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{settings.GCP_PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        logger.info(f"Successfully retrieved secret: {secret_id}")
        return secret_value
    except NotFound:
        logger.warning(f"Secret not found: {secret_id}")
    except PermissionDenied:
        logger.warning(f"Permission denied for secret: {secret_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch secret {secret_id}: {e}")
    return None
