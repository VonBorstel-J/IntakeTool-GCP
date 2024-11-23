#gemini_utils.py
import os, yaml, logging, asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.cloud import aiplatform
from fastapi import HTTPException
from fastapi.responses import JSONResponse

load_dotenv()
logger = logging.getLogger("gemini_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
ERROR_CODES = {"gemini_api_error": "ERR005", "prompt_error": "ERR006", "parse_error": "ERR002", "server_error": "ERR004"}

def load_prompts(file_path: str = "prompts.yaml") -> Dict[str, str]:
    try:
        with open(file_path, 'r') as file:
            prompts = yaml.safe_load(file)
            logger.info("Successfully loaded prompt templates from prompts.yaml.")
            return prompts
    except FileNotFoundError:
        logger.error(f"{ERROR_CODES['prompt_error']}: prompts.yaml file not found.")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['prompt_error'], "message": "Prompt configuration file not found."}})
    except yaml.YAMLError as e:
        logger.error(f"{ERROR_CODES['prompt_error']}: YAML parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['prompt_error'], "message": "Error parsing prompt configuration."}})

PROMPTS = load_prompts()

def generate_prompt(template_name: str, context: Dict[str, Any]) -> str:
    try:
        template = PROMPTS.get(template_name)
        if not template:
            logger.error(f"{ERROR_CODES['prompt_error']}: Template '{template_name}' not found.")
            raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['prompt_error'], "message": f"Prompt template '{template_name}' not found."}})
        prompt = template.format(**context)
        logger.info(f"Generated prompt using template '{template_name}'.")
        return prompt
    except KeyError as e:
        logger.error(f"{ERROR_CODES['prompt_error']}: Missing key in context for prompt generation: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['prompt_error'], "message": "Incomplete context for prompt generation."}})
    except Exception as e:
        logger.error(f"{ERROR_CODES['prompt_error']}: Unexpected error during prompt generation: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['prompt_error'], "message": "Error generating prompt."}})

def initialize_vertex_ai():
    try:
        project = os.getenv("GCP_PROJECT_ID", "your_project_id")
        location = os.getenv("GCP_LOCATION", "us-central1")  # Ensure this matches a supported region
        if os.getenv("MOCK_VERTEX_AI", "false").lower() == "true":
            logger.warning("Vertex AI is mocked for development.")
            return
        aiplatform.init(project=project, location=location)
        logger.info("Initialized Vertex AI with project and location.")
    except Exception as e:
        logger.error(f"{ERROR_CODES['gemini_api_error']}: Failed to initialize Vertex AI. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['gemini_api_error'], "message": "Failed to initialize AI platform."}})


initialize_vertex_ai()

@retry(retry=retry_if_exception_type((GoogleAPICallError, RetryError)), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def call_gemini_api(prompt: str) -> Dict[str, Any]:
    try:
        model_name = os.getenv("GEMINI_MODEL_NAME", "your_gemini_model_name")
        endpoint = os.getenv("GEMINI_ENDPOINT", "projects/your_project_id/locations/your_location/endpoints/your_endpoint_id")
        predictor = aiplatform.PipelineServiceClient()
        response = await asyncio.to_thread(predictor.predict, endpoint=endpoint, instances=[{"prompt": prompt}])
        logger.info("Successfully called Gemini API.")
        return response
    except GoogleAPICallError as e:
        logger.error(f"{ERROR_CODES['gemini_api_error']}: Google API call error: {str(e)}")
        raise HTTPException(status_code=502, detail={"error": {"code": ERROR_CODES['gemini_api_error'], "message": "Gemini API call failed."}})
    except Exception as e:
        logger.error(f"{ERROR_CODES['gemini_api_error']}: Unexpected error during Gemini API call: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['gemini_api_error'], "message": "Error communicating with Gemini API."}})

def parse_email_content(email_content: Dict[str, Any]) -> str:
    try:
        context = {
            "subject": email_content.get("subject", ""),
            "body": email_content.get("body", ""),
            "attachments_info": ", ".join([att['filename'] for att in email_content.get('attachments', []) if att.get('filename')])
        }
        prompt = generate_prompt("email_parsing", context)
        return prompt
    except Exception as e:
        logger.error(f"{ERROR_CODES['parse_error']}: Failed to generate prompt from email content. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['parse_error'], "message": "Failed to generate prompt from email content."}})
