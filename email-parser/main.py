# main.py
import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Dict, List
from gmail_utils import fetch_email, fetch_email_batch
from gemini_utils import parse_email_content, call_gemini_api
from google.cloud import bigquery
from datetime import datetime

import asyncio

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_file = os.getenv("LOG_FILE", "app.log")
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EmailRequest(BaseModel):
    email_id: str

class BatchEmailRequest(BaseModel):
    email_ids: List[str]

class CustomTextRequest(BaseModel):
    custom_text: str

# In-memory progress tracker
parsing_progress: Dict[str, Dict[str, Any]] = {}

def retry_on_exception():
    return retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)

TRIAL_GUIDE_URL = "https://example.com/troubleshooting"
ERROR_CODES = {
    "fetch_error": "ERR001",
    "parse_error": "ERR002",
    "invalid_input": "ERR003",
    "server_error": "ERR004"
}

def create_error_response(code: str, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "troubleshooting": TRIAL_GUIDE_URL
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Application is up and running.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down.")

@app.get("/api/status")
async def read_status():
    logger.info("Status check requested.")
    return {"status": "Service is up and running"}

@app.post("/api/parse-email")
@limiter.limit("5/minute")
@retry_on_exception()
async def parse_email(request: EmailRequest):
    email_id = request.email_id
    task_id = f"email_{email_id}"

    # Initialize progress tracking
    parsing_progress[task_id] = {
        "steps": ["Fetching Data", "Parsing Data", "Storing Data"],
        "current_step": 0,
        "status": "In Progress"
    }

    try:
        # Step 1: Fetch email
        email_content = await fetch_email(email_id)
        parsing_progress[task_id]["current_step"] = 1

        # Step 2: Parse email
        prompt = parse_email_content(email_content)
        parsed_data = await call_gemini_api(prompt)
        parsing_progress[task_id]["current_step"] = 2

        # Step 3: Store results in BigQuery
        await insert_raw_parsed_output(email_id, parsed_data)
        parsing_progress[task_id]["current_step"] = 3
        parsing_progress[task_id]["status"] = "Completed"

        # Return success response
        return {
            "email_id": email_id,
            "parsed_data": parsed_data,
            "status": "Success"
        }

    except HTTPException as he:
        logger.error(f"HTTPException at step {parsing_progress[task_id]['current_step']}: {he.detail}")
        parsing_progress[task_id]["status"] = "Failed"
        raise HTTPException(
            status_code=he.status_code,
            detail=create_error_response(
                ERROR_CODES['parse_error'],
                f"Operation failed at step {parsing_progress[task_id]['current_step']}: {he.detail}"
            )
        )
    except Exception as e:
        logger.error(f"Unexpected error at step {parsing_progress[task_id]['current_step']}: {str(e)}")
        parsing_progress[task_id]["status"] = "Failed"
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ERROR_CODES['server_error'],
                f"Unexpected error at step {parsing_progress[task_id]['current_step']}: {str(e)}"
            )
        )
    finally:
        # Cleanup progress tracking only if completed or failed
        if parsing_progress[task_id]["status"] in ["Completed", "Failed"]:
            parsing_progress.pop(task_id, None)

@app.post("/api/parse-text")
@limiter.limit("5/minute")
@retry_on_exception()
async def parse_text(request: CustomTextRequest):
    custom_text = request.custom_text
    task_id = f"text_{hash(custom_text)}"
    parsing_progress[task_id] = {
        "steps": ["Processing Text", "Parsing Data", "Finalizing"],
        "current_step": 0
    }
    try:
        logger.info("Received custom text for parsing.")
        parsing_progress[task_id]["current_step"] = 1
        parsed_data = await call_gemini_api(custom_text)
        parsing_progress[task_id]["current_step"] = 3
        logger.info("Successfully parsed custom text.")
        parsing_progress.pop(task_id, None)
        return {"parsed_data": parsed_data}
    except HTTPException as he:
        logger.error(f"{ERROR_CODES['parse_error']}: Parsing failed for custom text. Cause: {he.detail}")
        parsing_progress.pop(task_id, None)
        raise HTTPException(status_code=he.status_code, detail=create_error_response(ERROR_CODES['parse_error'], "Failed to parse custom text. Please check the input."))
    except Exception as e:
        logger.error(f"{ERROR_CODES['parse_error']}: Parsing failed for custom text. Cause: {str(e)}")
        parsing_progress.pop(task_id, None)
        raise HTTPException(status_code=500, detail=create_error_response(ERROR_CODES['parse_error'], "Failed to parse custom text due to a server error. Please try again later."))

@app.get("/api/progress")
async def get_progress(task_id: str = None):
    """
    Optional query parameter 'task_id' to get progress for a specific task.
    If no task_id is provided, return overall or mock progress.
    """
    if task_id:
        progress = parsing_progress.get(task_id)
        if not progress:
            return {"message": "No active task found for the provided task_id."}
        return progress
    else:
        # Mock progress if task_id is not provided
        return {
            "steps": ["Fetching Data", "Parsing Data", "Finalizing"],
            "current_step": 2
        }

@app.post("/api/process-batch")
@limiter.limit("2/minute")
@retry_on_exception()
async def process_batch(request: BatchEmailRequest):
    email_ids = request.email_ids
    if not email_ids:
        logger.error(f"{ERROR_CODES['invalid_input']}: No email IDs provided in batch request.")
        raise HTTPException(status_code=400, detail=create_error_response(ERROR_CODES['invalid_input'], "No email IDs provided for batch processing."))
    try:
        logger.info(f"Fetching batch of {len(email_ids)} emails.")
        emails = await fetch_email_batch(email_ids)
        logger.info(f"Successfully fetched {len(emails)} emails.")
    except HTTPException as he:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch batch of emails. Cause: {he.detail}")
        raise HTTPException(status_code=he.status_code, detail=create_error_response(ERROR_CODES['fetch_error'], "Failed to fetch one or more emails. Please verify the email IDs."))
    except Exception as e:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch batch of emails. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail=create_error_response(ERROR_CODES['fetch_error'], "Failed to fetch emails due to a server error. Please try again later."))
    results = []
    for email in emails:
        email_id = email.get('message_id')
        task_id = f"email_{email_id}"
        parsing_progress[task_id] = {
            "steps": ["Fetching Data", "Parsing Data", "Finalizing"],
            "current_step": 0
        }
        if not email_id:
            logger.warning(f"{ERROR_CODES['parse_error']}: Missing message_id in fetched email data.")
            results.append({"email_id": None, "error": create_error_response(ERROR_CODES['parse_error'], "Missing message ID in fetched email data.")})
            parsing_progress.pop(task_id, None)
            continue
        try:
            logger.info(f"Parsing email content for ID: {email_id}")
            prompt = parse_email_content(email)
            parsing_progress[task_id]["current_step"] = 1
            parsed_data = await call_gemini_api(prompt)
            parsing_progress[task_id]["current_step"] = 3
            results.append({"email_id": email_id, "parsed_data": parsed_data})
            logger.info(f"Successfully parsed email with ID: {email_id}")
            parsing_progress.pop(task_id, None)
        except Exception as e:
            logger.error(f"{ERROR_CODES['parse_error']}: Parsing failed for email ID {email_id}. Cause: {str(e)}")
            results.append({"email_id": email_id, "error": create_error_response(ERROR_CODES['parse_error'], f"Failed to parse email ID {email_id}.")})
            parsing_progress.pop(task_id, None)
    return {"results": results}

async def insert_raw_parsed_output(email_id: str, raw_output: str, parser_version="1.0"):
    client = bigquery.Client()
    table_id = "forensicemailparser:email_parsing_dataset.raw_parsed_output"

    rows_to_insert = [{
        "email_id": email_id,
        "parsed_timestamp": datetime.utcnow().isoformat(),  
        "raw_parsed_output": raw_output,
        "parser_version": parser_version
    }]

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, client.insert_rows_json, table_id, rows_to_insert)
        logger.info(f"Successfully inserted raw parsed output for email ID: {email_id}")
    except Exception as e:
        logger.error(f"Failed to insert raw parsed output for email ID {email_id}: {str(e)}. Data: {rows_to_insert}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"{ERROR_CODES['server_error']}: Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content=create_error_response(ERROR_CODES['server_error'], "An internal server error occurred. Please try again later."))

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    try:
        # Serve the index.html file from the static directory
        return FileResponse(os.path.join("static", "index.html"))
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Error serving frontend: {e}")
        raise HTTPException(status_code=500, detail="Failed to load the frontend application.")
    
def get_settings() -> Dict[str, Any]:
    return {"some_setting": os.getenv("SOME_SETTING", "default_value")}
