# gemini_utils.py

import os
import yaml
import asyncio
import logging
from typing import Dict, Any, List, Optional
from google.api_core.exceptions import GoogleAPICallError
from config import settings, get_logger
from exceptions import GeminiAPIError, PromptError, ParseError
import aiohttp
import time
import json

logger = get_logger({"module": "gemini_utils"})

def load_prompts(file_path: str = "prompts.yaml") -> Dict[str, Any]:
    """
    Load prompt templates and configurations from a YAML file and validate them.
    """
    try:
        with open(file_path, 'r') as file:
            prompts = yaml.safe_load(file) or {}
            # Validate each prompt template
            for section, templates in prompts.items():
                if isinstance(templates, dict):
                    for template_name, template_content in templates.items():
                        if isinstance(template_content, str):
                            # Example: Ensure placeholders are present if required
                            required_keys = get_required_keys_for_template(template_name)
                            validate_prompt_template(template_content, required_keys)
                elif isinstance(templates, list):
                    for item in templates:
                        # Handle list-based configurations if any
                        pass
            logger.info(f"Loaded and validated {len(prompts)} prompt sections from {file_path}.")
            return prompts
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {file_path}")
        raise PromptError(f"Prompts file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {file_path}: {e}")
        raise PromptError(f"YAML parsing error: {e}")
    except PromptError as e:
        logger.error(f"Prompt validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading prompts: {e}", exc_info=True)
        raise PromptError("Failed to load prompts due to an unexpected error.") from e

def get_required_keys_for_template(template_name: str) -> List[str]:
    """
    Define required placeholders for each template.
    """
    required_keys_map = {
        "email_parsing": ["email_thread_content", "attachments_content"],
        # Add other templates and their required keys here
    }
    return required_keys_map.get(template_name, [])

def validate_prompt_template(template: str, required_keys: List[str]) -> None:
    """
    Ensure that the prompt template contains all required placeholders.
    """
    missing_keys = [key for key in required_keys if f"{{{{{key}}}}}" not in template]
    if missing_keys:
        warning_msg = f"Template missing required keys: {', '.join(missing_keys)}"
        logger.warning(warning_msg)
        # Do not raise exception to prevent app crash; just log warning

def validate_prompt_length(prompt: str, max_length: int = 5000) -> None:
    """
    Ensure the prompt does not exceed the maximum allowed length.
    """
    if len(prompt) > max_length:
        logger.warning(f"Prompt length {len(prompt)} exceeds maximum of {max_length} characters.")
        # Optionally, you can trim the prompt or handle accordingly

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
        logger.warning("Prompt template not found.", extra={"template_name": template_name})
        raise PromptError(f"Prompt template '{template_name}' not found.")
    
    try:
        sanitized_context = sanitize_context(context)
        prompt = template.format(**sanitized_context)
        validate_prompt_length(prompt, max_length=get_max_tokens_for_prompt(template_name))
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

def get_max_tokens_for_prompt(template_name: str) -> int:
    """
    Retrieve the maximum token limit for a given prompt from prompts.yaml.
    """
    try:
        max_tokens = PROMPTS['ai']['generative_ai']['google']['max_tokens']
        dynamic_adjustment = PROMPTS.get('dynamic_token_adjustment', {}).get('enabled', False)
        if dynamic_adjustment:
            threshold = PROMPTS['dynamic_token_adjustment'].get('max_tokens_threshold', 2500)
            return min(max_tokens, threshold)
        return max_tokens
    except KeyError:
        logger.warning(f"Max tokens not defined for template '{template_name}'. Using default 5000.")
        return 5000  # Default value if not specified

async def call_gemini_api(prompt: str, retries: int = 3, backoff_factor: float = 0.5) -> Dict[str, Any]:
    """
    Asynchronously call the Gemini API with the provided prompt using HTTP requests.
    Implements retry mechanism for transient errors.
    """
    mock_vertex_ai = os.getenv("MOCK_VERTEX_AI", "false").lower() == "true"
    if mock_vertex_ai:
        logger.info("MOCK_VERTEX_AI enabled. Returning mock response.")
        return {"predictions": ["Mock response"]}
    
    # Validate GEMINI_API_KEY
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        logger.warning("GEMINI_API_KEY is missing.")
        raise GeminiAPIError("GEMINI_API_KEY is missing.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {}
    }
    
    attempt = 0
    while attempt <= retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(settings.GEMINI_ENDPOINT, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Gemini API call successful.", extra={"response": result})
                        return result
                    elif response.status in {500, 502, 503, 504}:
                        # Transient server errors
                        logger.warning(f"Gemini API transient error {response.status}. Retrying...", extra={"attempt": attempt + 1})
                        raise GeminiAPIError(f"Transient error {response.status}")
                    else:
                        # Non-retriable errors
                        error_text = await response.text()
                        logger.error("Gemini API call failed.", extra={"status": response.status, "error": error_text})
                        raise GeminiAPIError(f"Gemini API call failed with status {response.status}: {error_text}")
        except (aiohttp.ClientError, asyncio.TimeoutError, GeminiAPIError) as e:
            attempt += 1
            if attempt > retries:
                logger.error("Exceeded maximum retries for Gemini API call.", extra={"error": str(e)})
                raise GeminiAPIError("Exceeded maximum retries for Gemini API call.") from e
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"Retrying Gemini API call in {sleep_time} seconds...", extra={"attempt": attempt})
            await asyncio.sleep(sleep_time)
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
            "email_thread_content": email_content.get("body", ""),
            "attachments_content": extract_attachments_content(email_content.get("attachments", []))
        }
        prompt = generate_prompt("parser.email_parsing", context)
        logger.debug("Email content parsed and prompt generated.", extra={"context": context})
        return prompt
    except PromptError as e:
        logger.error("Failed to generate prompt from email content.", extra={"error": str(e)})
        raise ParseError("Failed to generate prompt from email content.") from e

def extract_attachments_content(attachments: List[Dict[str, Any]]) -> str:
    """
    Extract and concatenate the content of all attachments.
    """
    contents = []
    for att in attachments:
        content = att.get('extracted_data_from_attachments')
        if content:
            contents.append(content)
    return "\n".join(contents)

# ------------------------- Batch Processing Functions ----------------------

async def batch_process_emails(emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes emails in batches as defined in prompts.yaml.
    Ensures max_tokens limit is respected.
    """
    batch_size = settings.batch_processing.get('batch_size', 100)
    batches = [emails[i:i + batch_size] for i in range(0, len(emails), batch_size)]
    processed_outputs = []

    for batch in batches:
        combined_content = "\n".join([email.get('body', '') for email in batch])
        attachments_content = "\n".join([
            att.get('extracted_data_from_attachments', '') for email in batch for att in email.get('attachments', [])
        ])
        context = {
            "email_thread_content": combined_content,
            "attachments_content": attachments_content
        }
        prompt = generate_prompt("parser.email_parsing", context)
        try:
            response = await call_gemini_api(prompt)
            parsed_data = response.get('predictions', [])
            # Assuming the Gemini API returns a list of parsed JSON strings
            for data in parsed_data:
                parsed_json = json.loads(data)
                processed_outputs.append(parsed_json)
        except ParseError as pe:
            logger.error(f"ParseError during batch processing: {pe}")
            continue
        except GeminiAPIError as ge:
            logger.error(f"GeminiAPIError during batch processing: {ge}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error during batch processing: {e}", exc_info=True)
            continue

    return processed_outputs

# ------------------------- Initialize Gemini Utils ------------------------

def initialize_gemini_utils():
    """
    Initialize and validate Gemini Utils on app startup.
    """
    # Validate GEMINI_API_KEY
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. API calls will fail.")
    
    # Validate prompts
    try:
        load_prompts()
    except PromptError as e:
        logger.warning(f"Prompt loading issues detected: {e}")
        # Continue without crashing

# Ensure initialization runs on module load
initialize_gemini_utils()

# ------------------------- End of File -------------------------------------
