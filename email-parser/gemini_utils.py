#gemini_utils.py
import os
import yaml
import asyncio
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import aiplatform
from config import get_logger, settings
from exceptions import GeminiAPIError, PromptError, ParseError

logger = get_logger({"module": "gemini_utils"})

def load_prompts(file_path: str = "prompts.yaml") -> Dict[str, str]:
    try:
        with open(file_path, 'r') as file:
            prompts = yaml.safe_load(file)
            for template_name, template in prompts.items():
                validate_prompt_template(template, required_keys=[])
            return prompts
    except Exception as e:
        logger.error(f"Error loading prompts from {file_path}. Error: {e}")
        raise PromptError("Failed to load prompts.")

def validate_prompt_template(template: str, required_keys: List[str]):
    for key in required_keys:
        if f"{{{{{key}}}}}" not in template:
            raise PromptError(f"Template missing required key: {key}")

def validate_prompt_length(prompt: str, max_length: int = 5000) -> None:
    if len(prompt) > max_length:
        logger.error(f"Prompt length {len(prompt)} exceeds maximum allowed length {max_length}.")
        raise PromptError("Prompt exceeds maximum length.")

PROMPTS = load_prompts()

def generate_prompt(template_name: str, context: Dict[str, Any]) -> str:
    template = PROMPTS.get(template_name)
    if not template:
        logger.error(f"Template '{template_name}' not found.")
        raise PromptError(f"Prompt template '{template_name}' not found.")
    try:
        prompt = template.format(**context)
        validate_prompt_length(prompt)
        return prompt
    except KeyError as e:
        logger.error(f"Missing key in context for template '{template_name}'. Context: {context}, Error: {e}")
        raise PromptError(f"Missing key in context: {e}")
    except Exception as e:
        logger.error(f"Error generating prompt. Template: {template_name}, Context: {context}, Error: {e}")
        raise PromptError("Error generating prompt.")

def get_endpoint():
    endpoint = None
    def _get_endpoint():
        nonlocal endpoint
        if endpoint is not None:
            return endpoint
        if os.getenv("MOCK_VERTEX_AI", "false").lower() == "true":
            logger.warning("Vertex AI is mocked for development.")
            return None
        try:
            aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
            endpoint_instance = aiplatform.Endpoint(endpoint_name=settings.GEMINI_ENDPOINT)
            logger.info("Initialized Vertex AI endpoint.")
            endpoint = endpoint_instance
            return endpoint
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI. Error: {e}")
            raise GeminiAPIError("Failed to initialize AI platform.")
    return _get_endpoint

get_endpoint = get_endpoint()

@retry(
    retry=retry_if_exception_type(GoogleAPICallError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def call_gemini_api(prompt: str) -> Dict[str, Any]:
    endpoint = get_endpoint()
    if endpoint is None:
        if os.getenv("MOCK_VERTEX_AI", "false").lower() == "true":
            logger.info("MOCK_VERTEX_AI is enabled. Returning mock response.")
            return {"predictions": ["Mock response"]}
        else:
            logger.error("Endpoint is not initialized.")
            raise GeminiAPIError("Endpoint is not initialized.")
    try:
        # Expected response structure: {"predictions": [...]}
        response = await asyncio.to_thread(endpoint.predict, instances=[{"prompt": prompt}])
        return response
    except GoogleAPICallError as e:
        logger.error(f"Google API call error during Gemini API call. Error: {e}")
        raise GeminiAPIError("Gemini API call failed.")
    except Exception as e:
        logger.error(f"Error during Gemini API call. Error: {e}")
        raise GeminiAPIError("Error communicating with Gemini API.")

def parse_email_content(email_content: Dict[str, Any]) -> str:
    try:
        context = {
            "subject": email_content.get("subject", ""),
            "body": email_content.get("body", ""),
            "attachments_info": ", ".join(
                att.get('filename', '') for att in email_content.get('attachments', [])
            )
        }
        prompt = generate_prompt("email_parsing", context)
        return prompt
    except PromptError as e:
        logger.error(f"Error generating prompt from email content. Error: {e}")
        raise ParseError("Failed to generate prompt from email content.")
