# gemini_utils.py

import os
import yaml
import asyncio
from typing import Dict, Any, List, Optional
from google.api_core.exceptions import GoogleAPICallError
from config import settings, get_logger
from exceptions import GeminiAPIError, PromptError, ParseError
import aiohttp

logger = get_logger({"module": "gemini_utils"})

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
    Asynchronously call the Gemini API with the provided prompt using HTTP requests.
    """
    mock_vertex_ai = os.getenv("MOCK_VERTEX_AI", "false").lower() == "true"
    if mock_vertex_ai:
        logger.info("MOCK_VERTEX_AI enabled. Returning mock response.")
        return {"predictions": ["Mock response"]}
    
    headers = {
        "Authorization": f"Bearer {settings.GEMINI_API_KEY.get_secret_value()}",
        "Content-Type": "application/json",
    }
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.GEMINI_ENDPOINT, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Gemini API call failed.", extra={"status": response.status, "error": error_text})
                    raise GeminiAPIError(f"Gemini API call failed with status {response.status}: {error_text}")
                result = await response.json()
                logger.debug("Gemini API call successful.", extra={"response": result})
                return result
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
            "subject": email_content.get("metadata", {}).get("subject", ""),
            "body": email_content.get("body", ""),
            "attachments_info": ", ".join(
                att.get('attachment_name', '') for att in email_content.get('attachments', [])
            )
        }
        prompt = generate_prompt("email_parsing", context)
        logger.debug("Email content parsed and prompt generated.", extra={"context": context})
        return prompt
    except PromptError as e:
        logger.error("Failed to generate prompt from email content.", extra={"error": str(e)})
        raise ParseError("Failed to generate prompt from email content.") from e
