#main.py
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Any, Dict, List
from gmail_utils import fetch_email, fetch_email_batch, GmailUtilsException
from gemini_utils import parse_email_content, call_gemini_api
from google.cloud import bigquery
from datetime import datetime
import asyncio
from config import get_logger, settings
from exceptions import FetchError, ParseError, ServerError, InvalidInputError
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(status_code=429, content={"error": {"code": "ERR_RATE_LIMIT", "message": "Rate limit exceeded", "troubleshooting": "https://example.com/troubleshooting"}}))
logger = get_logger()
TRIAL_GUIDE_URL = "https://example.com/troubleshooting"
ERROR_CODES = {"fetch_error": "ERR001", "parse_error": "ERR002", "invalid_input": "ERR003", "server_error": "ERR004"}

class EmailRequest(BaseModel):
    email_id: str

class BatchEmailRequest(BaseModel):
    email_ids: List[str]

class CustomTextRequest(BaseModel):
    custom_text: str

class ProgressTracker:
    def __init__(self):
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def set_progress(self, task_id: str, progress: Dict[str, Any]):
        async with self._lock:
            self._progress[task_id] = progress
    
    async def update_step(self, task_id: str, step: int):
        async with self._lock:
            if task_id in self._progress:
                self._progress[task_id]["current_step"] = step
    
    async def set_status(self, task_id: str, status: str):
        async with self._lock:
            if task_id in self._progress:
                self._progress[task_id]["status"] = status
    
    async def get_progress(self, task_id: str):
        async with self._lock:
            return self._progress.get(task_id)
    
    async def remove_progress(self, task_id: str):
        async with self._lock:
            self._progress.pop(task_id, None)

progress_tracker = ProgressTracker()

def create_error_response(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "troubleshooting": TRIAL_GUIDE_URL}}

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

def get_retry():
    return AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type((FetchError, ParseError, ServerError)))

@app.post("/api/parse-email")
@limiter.limit("5/minute")
async def parse_email(request: EmailRequest):
    email_id = request.email_id
    task_id = f"email_{email_id}"
    await progress_tracker.set_progress(task_id, {"steps": ["Fetching Data", "Parsing Data", "Storing Data"], "current_step": 0, "status": "In Progress"})
    async for attempt in get_retry():
        with attempt:
            try:
                email_content = await fetch_email(email_id)
                await progress_tracker.update_step(task_id, 1)
                prompt = parse_email_content(email_content)
                parsed_data = await call_gemini_api(prompt)
                await progress_tracker.update_step(task_id, 2)
                await insert_raw_parsed_output(email_id, parsed_data)
                await progress_tracker.update_step(task_id, 3)
                await progress_tracker.set_status(task_id, "Completed")
                return {"email_id": email_id, "parsed_data": parsed_data, "status": "Success"}
            except (FetchError, ParseError, ServerError) as e:
                await progress_tracker.set_status(task_id, "Failed")
                raise e
    await progress_tracker.remove_progress(task_id)

@app.post("/api/parse-text")
@limiter.limit("5/minute")
async def parse_text(request: CustomTextRequest):
    custom_text = request.custom_text
    task_id = f"text_{hash(custom_text)}"
    await progress_tracker.set_progress(task_id, {"steps": ["Processing Text", "Parsing Data", "Finalizing"], "current_step": 0})
    async for attempt in get_retry():
        with attempt:
            try:
                await progress_tracker.update_step(task_id, 1)
                parsed_data = await call_gemini_api(custom_text)
                await progress_tracker.update_step(task_id, 3)
                await progress_tracker.remove_progress(task_id)
                return {"parsed_data": parsed_data}
            except (ParseError, ServerError) as e:
                await progress_tracker.remove_progress(task_id)
                raise e
    return {"parsed_data": None}

@app.get("/api/progress")
async def get_progress(task_id: str = None):
    if task_id:
        progress = await progress_tracker.get_progress(task_id)
        if not progress:
            return {"message": "No active task found for the provided task_id."}
        return progress
    return {"steps": ["Fetching Data", "Parsing Data", "Finalizing"], "current_step": 2}

@app.post("/api/process-batch")
@limiter.limit("2/minute")
async def process_batch(request: BatchEmailRequest):
    email_ids = request.email_ids
    if not email_ids:
        logger.error(f"{ERROR_CODES['invalid_input']}: No email IDs provided in batch request.")
        raise HTTPException(status_code=400, detail=create_error_response(ERROR_CODES['invalid_input'], "No email IDs provided for batch processing."))
    try:
        emails = await fetch_email_batch(email_ids)
    except FetchError as fe:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch batch of emails. Cause: {fe}")
        raise HTTPException(status_code=500, detail=create_error_response(ERROR_CODES['fetch_error'], "Failed to fetch one or more emails. Please verify the email IDs."))
    except ServerError as se:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch batch of emails. Cause: {se}")
        raise HTTPException(status_code=500, detail=create_error_response(ERROR_CODES['fetch_error'], "Failed to fetch emails due to a server error. Please try again later."))
    tasks = []
    results = []
    for email in emails:
        email_id = email.get('message_id')
        if not email_id:
            results.append({"email_id": None, "error": create_error_response(ERROR_CODES['parse_error'], "Missing message ID in fetched email data.")})
            continue
        tasks.append(process_single_email(email_id, email, results))
    await asyncio.gather(*tasks)
    return {"results": results}

async def process_single_email(email_id: str, email: Dict[str, Any], results: List[Dict[str, Any]]):
    task_id = f"email_{email_id}"
    await progress_tracker.set_progress(task_id, {"steps": ["Fetching Data", "Parsing Data", "Finalizing"], "current_step": 0, "status": "In Progress"})
    async for attempt in get_retry():
        with attempt:
            try:
                prompt = parse_email_content(email)
                await progress_tracker.update_step(task_id, 1)
                parsed_data = await call_gemini_api(prompt)
                await progress_tracker.update_step(task_id, 3)
                results.append({"email_id": email_id, "parsed_data": parsed_data, "status": "Success"})
                await progress_tracker.remove_progress(task_id)
            except (ParseError, ServerError) as e:
                results.append({"email_id": email_id, "error": create_error_response(ERROR_CODES['server_error'], f"Failed to parse email ID {email_id} due to a server error.")})
                await progress_tracker.remove_progress(task_id)
                break

async def insert_raw_parsed_output(email_id: str, raw_output: str, parser_version="1.0"):
    client = bigquery.Client()
    table_id = f"{settings.BIGQUERY_PROJECT_ID}.{settings.BIGQUERY_DATASET}.raw_parsed_output"
    rows_to_insert = [{"email_id": email_id, "parsed_timestamp": datetime.utcnow().isoformat(), "raw_parsed_output": raw_output, "parser_version": parser_version}]
    try:
        await asyncio.to_thread(client.insert_rows_json, table_id, rows_to_insert)
    except Exception as e:
        logger.error(f"Failed to insert raw parsed output for email ID {email_id}: {str(e)}. Data: {rows_to_insert}")
        raise ServerError(f"Failed to insert parsed data for email ID {email_id}.")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"{ERROR_CODES['server_error']}: Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content=create_error_response(ERROR_CODES['server_error'], "An internal server error occurred. Please try again later."))

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))
