# gemini_utils.py

import os
import yaml
import asyncio
from typing import Dict, Any, List, Optional
from google.api_core.retry import Retry
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import aiplatform
from config import settings, get_logger
from exceptions import GeminiAPIError, PromptError, ParseError

logger = get_logger({"module": "gemini_utils"})

# Initialize GCP Clients Globally
class GCPClients:
    vertex_ai_endpoint: Optional[aiplatform.Endpoint] = None

    @classmethod
    async def initialize_vertex_ai_endpoint(cls):
        mock_vertex_ai = os.getenv("MOCK_VERTEX_AI", "false").lower() == "true"
        if mock_vertex_ai:
            logger.warning("Vertex AI is mocked for development.")
            cls.vertex_ai_endpoint = None
            return
        try:
            aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
            cls.vertex_ai_endpoint = aiplatform.Endpoint(endpoint_name=settings.GEMINI_ENDPOINT)
            logger.info("Vertex AI endpoint initialized successfully.", extra={"endpoint": settings.GEMINI_ENDPOINT})
        except Exception as e:
            logger.error("Failed to initialize Vertex AI endpoint.", exc_info=True, extra={"error": str(e)})
            raise GeminiAPIError("Failed to initialize Vertex AI endpoint.") from e

# Initialize Vertex AI Endpoint on module load
asyncio.create_task(GCPClients.initialize_vertex_ai_endpoint())

def load_prompts(file_path: str = "prompts.yaml") -> Dict[str, str]:
    """
    Load prompt templates from a YAML file and validate them.
    """
    try:
        with open(file_path, 'r') as file:
            prompts = yaml.safe_load(file) or {}
            for template_name, template in prompts.items():
                validate_prompt_template(template, required_keys=[])
            logger.info(f"Loaded {len(prompts)} prompt templates from {file_path}.")
            return prompts
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {file_path}")
        raise PromptError(f"Prompts file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {file_path}: {e}")
        raise PromptError(f"YAML parsing error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading prompts: {e}", exc_info=True)
        raise PromptError("Failed to load prompts due to an unexpected error.") from e

def validate_prompt_template(template: str, required_keys: List[str]) -> None:
    """
    Ensure that the prompt template contains all required placeholders.
    """
    missing_keys = [key for key in required_keys if f"{{{{{key}}}}}" not in template]
    if missing_keys:
        error_msg = f"Template missing required keys: {', '.join(missing_keys)}"
        logger.error(error_msg)
        raise PromptError(error_msg)

def validate_prompt_length(prompt: str, max_length: int = 5000) -> None:
    """
    Ensure the prompt does not exceed the maximum allowed length.
    """
    if len(prompt) > max_length:
        logger.error(f"Prompt length {len(prompt)} exceeds maximum of {max_length} characters.")
        raise PromptError("Prompt exceeds the maximum allowed length.")

def sanitize_input(value: Any) -> Any:
    """
    Sanitize input to prevent injection attacks by escaping braces.
    """
    if isinstance(value, str):
        return value.replace("{", "{{").replace("}", "}}")
    return value

def sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize all values in the context dictionary.
    """
    return {key: sanitize_input(value) for key, value in context.items()}

PROMPTS = load_prompts()

def generate_prompt(template_name: str, context: Dict[str, Any]) -> str:
    """
    Generate a prompt by formatting the specified template with the provided context.
    """
    template = PROMPTS.get(template_name)
    if not template:
        logger.error("Prompt template not found.", extra={"template_name": template_name})
        raise PromptError(f"Prompt template '{template_name}' not found.")
    
    try:
        sanitized_context = sanitize_context(context)
        prompt = template.format(**sanitized_context)
        validate_prompt_length(prompt)
        logger.debug("Prompt generated successfully.", extra={"template_name": template_name})
        return prompt
    except KeyError as e:
        logger.error("Missing key in context for prompt generation.", extra={
            "template_name": template_name,
            "missing_key": str(e)
        })
        raise PromptError(f"Missing key in context: {e}") from e
    except Exception as e:
        logger.error("Error during prompt generation.", extra={
            "template_name": template_name,
            "error": str(e)
        })
        raise PromptError("Error generating prompt.") from e

async def call_gemini_api(prompt: str) -> Dict[str, Any]:
    """
    Asynchronously call the Gemini API with the provided prompt.
    Utilizes asyncio.to_thread to prevent blocking the event loop.
    """
    mock_vertex_ai = os.getenv("MOCK_VERTEX_AI", "false").lower() == "true"
    if GCPClients.vertex_ai_endpoint is None and not mock_vertex_ai:
        logger.error("Vertex AI endpoint is not initialized.")
        raise GeminiAPIError("Vertex AI endpoint is not initialized.")
    
    if mock_vertex_ai:
        logger.info("MOCK_VERTEX_AI enabled. Returning mock response.")
        return {"predictions": ["Mock response"]}
    
    try:
        # Define a retry strategy using Vertex AI client's built-in retries
        retry_strategy = Retry(
            initial=1.0,
            maximum=10.0,
            multiplier=2.0,
            deadline=30.0,
            predicate=lambda exc: isinstance(exc, GoogleAPICallError)
        )
        logger.debug("Calling Vertex AI endpoint.", extra={"action": "Gemini API call"})
        response = await asyncio.to_thread(
            GCPClients.vertex_ai_endpoint.predict,
            instances=[{"prompt": prompt}],
            retry=retry_strategy
        )
        logger.debug("Gemini API call successful.", extra={"response": response})
        return response
    except GoogleAPICallError as e:
        logger.error("Google API call error during Gemini API interaction.", extra={"error": str(e)})
        raise GeminiAPIError("Gemini API call failed due to a Google API error.") from e
    except Exception as e:
        logger.error("Unexpected error during Gemini API interaction.", extra={"error": str(e)})
        raise GeminiAPIError("An unexpected error occurred while communicating with Gemini API.") from e

def parse_email_content(email_content: Dict[str, Any]) -> str:
    """
    Parse email content and generate a prompt for the Gemini API.
    """
    try:
        context = {
            "subject": email_content.get("subject", ""),
            "body": email_content.get("body", ""),
            "attachments_info": ", ".join(
                att.get('filename', '') for att in email_content.get('attachments', [])
            )
        }
        prompt = generate_prompt("email_parsing", context)
        logger.debug("Email content parsed and prompt generated.", extra={"context": context})
        return prompt
    except PromptError as e:
        logger.error("Failed to generate prompt from email content.", extra={"error": str(e)})
        raise ParseError("Failed to generate prompt from email content.") from e
