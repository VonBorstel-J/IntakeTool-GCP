# main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Any, Dict, List, Optional
from gmail_utils import (
    fetch_email,
    fetch_email_batch,
    list_emails_by_label,
    process_emails_by_label,
    clear_email_cache,
    get_cache_info
)
from gemini_utils import (
    parse_email_content,
    call_gemini_api
)
from google.cloud import bigquery
from datetime import datetime
import asyncio
from exceptions import FetchError, ParseError, ServerError, InvalidInputError
from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import secrets  

# ---------------------- Configuration Management ----------------------

class Settings(BaseSettings):
    GOOGLE_CLIENT_SECRETS_FILE: str = Field(..., env='GOOGLE_CLIENT_SECRETS_FILE')
    GOOGLE_SCOPES: List[str] = Field(default=["https://www.googleapis.com/auth/gmail.readonly"])
    BIGQUERY_PROJECT_ID: str = Field(..., env='BIGQUERY_PROJECT_ID')
    BIGQUERY_DATASET: str = Field(..., env='BIGQUERY_DATASET')
    SECRET_KEY: str = Field(..., env='SECRET_KEY')  # For session management
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env='ALLOWED_HOSTS')  # Adjust as needed
    REDIRECT_URI: str = Field(..., env='REDIRECT_URI')  # OAuth2 Redirect URI

    class Config:
        env_file = ".env"

settings = Settings()

# -------------------------- Logging Setup ----------------------------

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------------------- FastAPI App ----------------------------

app = FastAPI()

# Enforce HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# Session Middleware for In-Memory Token Storage
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# Rate Limiting Middleware
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "ERR_RATE_LIMIT",
                "message": "Rate limit exceeded",
                "troubleshooting": "https://example.com/troubleshooting"
            }
        }
    )

# ----------------------- Security Headers Middleware ------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --------------------- Exception Handling Middleware -------------------

class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as he:
            logger.warning(f"HTTPException: {he.detail}")
            return JSONResponse(
                status_code=he.status_code,
                content={"error": {"code": he.status_code, "message": he.detail, "troubleshooting": "https://example.com/troubleshooting"}}
            )
        except (FetchError, ParseError, ServerError, InvalidInputError) as custom_exc:
            error_code = getattr(custom_exc, 'code', 'ERR_UNKNOWN')
            logger.error(f"{error_code}: {str(custom_exc)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"code": error_code, "message": str(custom_exc), "troubleshooting": "https://example.com/troubleshooting"}}
            )
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "ERR_INTERNAL", "message": "An internal server error occurred. Please try again later.", "troubleshooting": "https://example.com/troubleshooting"}}
            )

app.add_middleware(ExceptionHandlingMiddleware)

# ------------------------- OAuth2 Configuration -----------------------

flow = Flow.from_client_secrets_file(
    settings.GOOGLE_CLIENT_SECRETS_FILE,
    scopes=settings.GOOGLE_SCOPES,
    redirect_uri=settings.REDIRECT_URI
)

# The OAuth2AuthorizationCodeBearer is not utilized in the current implementation.
# It can be removed or integrated as needed. For now, we'll remove it to avoid confusion.

# -------------------------- GCP Client Setup --------------------------

def get_bigquery_client():
    if not hasattr(app.state, 'bigquery_client'):
        app.state.bigquery_client = bigquery.Client(project=settings.BIGQUERY_PROJECT_ID)
    return app.state.bigquery_client

# --------------------------- Progress Tracker --------------------------

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

# ------------------------ Error Response Utility ------------------------

TRIAL_GUIDE_URL = "https://example.com/troubleshooting"
ERROR_CODES = {
    "fetch_error": "ERR001",
    "parse_error": "ERR002",
    "invalid_input": "ERR003",
    "server_error": "ERR004",
    "internal_error": "ERR_INTERNAL"
}

def create_error_response(code: str, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "troubleshooting": TRIAL_GUIDE_URL
        }
    }

# ---------------------------- Pydantic Models --------------------------

class EmailRequest(BaseModel):
    email_id: str

class BatchEmailRequest(BaseModel):
    email_ids: List[str]

class CustomTextRequest(BaseModel):
    custom_text: str

class LabelRequest(BaseModel):
    label_name: str

# -------------------------- Application Events -------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Application is up and running.")
    # Initialize BigQuery client
    app.state.bigquery_client = bigquery.Client(project=settings.BIGQUERY_PROJECT_ID)
    logger.info("BigQuery client initialized.")
    # Initialize OAuth2 flow's state token
    if not hasattr(app.state, 'oauth_flow'):
        app.state.oauth_flow = flow

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down.")
    # Clean up BigQuery client if necessary
    if hasattr(app.state, 'bigquery_client'):
        await asyncio.to_thread(app.state.bigquery_client.close)
        logger.info("BigQuery client closed.")

# --------------------------- OAuth Endpoints ---------------------------

@app.get("/api/login")
async def login(request: Request):
    # Generate a secure random state token for CSRF protection
    state = secrets.token_urlsafe(16)
    request.session['state'] = state
    authorization_url, _ = app.state.oauth_flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=state,
        prompt='consent'  # Ensures refresh token is received
    )
    logger.info("Initiated OAuth2 login.", extra={"authorization_url": authorization_url, "state": state})
    return RedirectResponse(url=authorization_url)

@app.get("/api/callback")
async def oauth_callback(request: Request):
    try:
        # Extract state and code from query parameters
        state = request.query_params.get("state")
        code = request.query_params.get("code")

        # Validate state parameter to prevent CSRF
        stored_state = request.session.get('state')
        if not state or not code or state != stored_state:
            logger.warning("Invalid state parameter or missing authorization code.")
            raise HTTPException(status_code=400, detail="Invalid state or authorization code missing.")

        # Fetch the token using the authorization code
        app.state.oauth_flow.fetch_token(code=code)

        creds = app.state.oauth_flow.credentials

        # Store credentials securely in session
        request.session['credentials'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }

        logger.info("OAuth2 authorization successful.", extra={"token_uri": creds.token_uri, "scopes": creds.scopes})

        return RedirectResponse(url="/")  # Redirect to frontend after successful login
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OAuth callback failed.")

@app.get("/api/logout")
async def logout(request: Request):
    request.session.pop('credentials', None)
    logger.info("User logged out successfully.")
    return RedirectResponse(url="/")  # Redirect to frontend after logout

# --------------------- Gmail Service Dependency -------------------------

async def get_gmail_credentials(request: Request) -> Credentials:
    credentials = request.session.get('credentials')
    if not credentials:
        logger.warning("User not authorized. Credentials not found in session.")
        raise HTTPException(status_code=401, detail="User not authorized. Please log in.")
    
    creds = Credentials(
        token=credentials['token'],
        refresh_token=credentials.get('refresh_token'),
        token_uri=credentials['token_uri'],
        client_id=credentials['client_id'],
        client_secret=credentials['client_secret'],
        scopes=credentials['scopes']
    )
    
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Update session with refreshed token
            request.session['credentials']['token'] = creds.token
            logger.info("Credentials refreshed successfully.")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}", exc_info=True)
            raise HTTPException(status_code=401, detail="Token refresh failed. Please log in again.")
    
    return creds

# ----------------------------- API Endpoints ----------------------------

@app.get("/api/status")
async def read_status():
    logger.info("Status check requested.")
    return {"status": "Service is up and running"}

@app.post("/api/parse-email")
@limiter.limit("5/minute")
async def parse_email_endpoint(
    request: EmailRequest,
    creds: Credentials = Depends(get_gmail_credentials),
    bigquery_client: bigquery.Client = Depends(get_bigquery_client)
):
    email_id = request.email_id
    task_id = f"email_{email_id}"
    await progress_tracker.set_progress(task_id, {
        "steps": ["Fetching Data", "Parsing Data", "Storing Data"],
        "current_step": 0,
        "status": "In Progress"
    })
    try:
        # Fetch Email Content
        email_content = await fetch_email(email_id, creds)
        await progress_tracker.update_step(task_id, 1)

        # Parse Email Content and Generate Prompt
        prompt = parse_email_content(email_content)

        # Call Gemini API with Prompt
        parsed_data = await call_gemini_api(prompt)
        await progress_tracker.update_step(task_id, 2)

        # Insert Parsed Data into BigQuery
        await insert_raw_parsed_output(email_id, parsed_data, bigquery_client)
        await progress_tracker.update_step(task_id, 3)

        # Update Task Status to Completed
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Email parsing completed for email ID: {email_id}")

        return {"email_id": email_id, "parsed_data": parsed_data, "status": "Success"}
    except (FetchError, ParseError, ServerError, InvalidInputError) as e:
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing email ID {email_id}: {e}")
        raise e
    finally:
        await progress_tracker.remove_progress(task_id)

@app.post("/api/parse-text")
@limiter.limit("5/minute")
async def parse_text_endpoint(request: CustomTextRequest):
    custom_text = request.custom_text
    task_id = f"text_{hash(custom_text)}"
    await progress_tracker.set_progress(task_id, {
        "steps": ["Processing Text", "Parsing Data", "Finalizing"],
        "current_step": 0,
        "status": "In Progress"
    })
    try:
        # Update Progress Step 1
        await progress_tracker.update_step(task_id, 1)

        # Call Gemini API with Custom Text
        parsed_data = await call_gemini_api(custom_text)
        await progress_tracker.update_step(task_id, 2)

        # Assuming no storage is needed for custom text parsing
        # Update Progress Step 3
        await progress_tracker.update_step(task_id, 3)

        # Update Task Status to Completed
        await progress_tracker.set_status(task_id, "Completed")
        logger.info("Custom text parsing completed.")

        return {"parsed_data": parsed_data, "status": "Success"}
    except (ParseError, ServerError, InvalidInputError) as e:
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing custom text: {e}")
        raise e
    finally:
        await progress_tracker.remove_progress(task_id)

@app.post("/api/process-batch")
@limiter.limit("2/minute")
async def process_batch_endpoint(
    request: BatchEmailRequest,
    creds: Credentials = Depends(get_gmail_credentials),
    bigquery_client: bigquery.Client = Depends(get_bigquery_client)
):
    email_ids = request.email_ids
    if not email_ids:
        logger.error(f"{ERROR_CODES['invalid_input']}: No email IDs provided in batch request.")
        raise HTTPException(
            status_code=400,
            detail=create_error_response(ERROR_CODES['invalid_input'], "No email IDs provided for batch processing.")
        )
    try:
        # Fetch Batch Emails
        emails = await fetch_email_batch(email_ids, creds)
    except FetchError as fe:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch batch of emails. Cause: {fe}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ERROR_CODES['fetch_error'],
                "Failed to fetch one or more emails. Please verify the email IDs."
            )
        )
    except ServerError as se:
        logger.error(f"{ERROR_CODES['server_error']}: Failed to fetch batch of emails. Cause: {se}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ERROR_CODES['server_error'],
                "Failed to fetch emails due to a server error. Please try again later."
            )
        )
    
    tasks = []
    results = []
    for email in emails:
        email_id = email.get('message_id')
        if not email_id:
            results.append({
                "email_id": None,
                "error": create_error_response(
                    ERROR_CODES['parse_error'],
                    "Missing message ID in fetched email data."
                )
            })
            continue
        tasks.append(process_single_email(email_id, email, results, creds, bigquery_client))
    
    await asyncio.gather(*tasks)
    logger.info(f"Batch processing completed for {len(email_ids)} emails.")
    return {"results": results}

@app.post("/api/parse-label")
@limiter.limit("5/minute")
async def parse_label_endpoint(
    request: LabelRequest,
    creds: Credentials = Depends(get_gmail_credentials),
    bigquery_client: bigquery.Client = Depends(get_bigquery_client)
):
    label_name = request.label_name
    task_id = f"label_{label_name}"
    await progress_tracker.set_progress(task_id, {
        "steps": ["Listing Emails", "Processing Emails", "Parsing Data", "Storing Data"],
        "current_step": 0,
        "status": "In Progress"
    })
    try:
        # Update Progress Step 1
        email_ids = await list_emails_by_label(label_name, creds)
        if not email_ids:
            logger.info(f"No emails found for label: {label_name}")
            await progress_tracker.set_status(task_id, "Completed")
            return {"label_name": label_name, "processed_emails": 0, "status": "No Emails Found"}

        await progress_tracker.update_step(task_id, 1)

        # Process Batch Emails
        emails = await fetch_email_batch(email_ids, creds)
        await progress_tracker.update_step(task_id, 2)

        # Process Each Email Individually
        for email in emails:
            email_id = email.get('message_id')
            if not email_id:
                results.append({
                    "email_id": None,
                    "error": create_error_response(
                        ERROR_CODES['parse_error'],
                        "Missing message ID in fetched email data."
                    )
                })
                continue
            # Process and Parse Email
            try:
                prompt = parse_email_content(email)
                parsed_data = await call_gemini_api(prompt)
                await insert_raw_parsed_output(email_id, parsed_data, bigquery_client)
                results.append({"email_id": email_id, "parsed_data": parsed_data, "status": "Success"})
            except (ParseError, ServerError, InvalidInputError) as e:
                results.append({
                    "email_id": email_id,
                    "error": create_error_response(
                        ERROR_CODES['server_error'],
                        f"Failed to parse email ID {email_id} due to a server error."
                    )
                })
                await progress_tracker.set_status(task_id, "Failed")
        
        await progress_tracker.update_step(task_id, 3)
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Label parsing completed for label: {label_name}")
        return {"label_name": label_name, "processed_emails": len(email_ids), "status": "Success"}
    except (FetchError, ParseError, ServerError, InvalidInputError) as e:
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing emails by label '{label_name}': {e}")
        raise e
    finally:
        await progress_tracker.remove_progress(task_id)

# ----------------------- Helper Functions -----------------------------

async def process_single_email(
    email_id: str,
    email: Dict[str, Any],
    results: List[Dict[str, Any]],
    creds: Credentials,
    bigquery_client: bigquery.Client
):
    task_id = f"email_{email_id}"
    await progress_tracker.set_progress(task_id, {
        "steps": ["Fetching Data", "Parsing Data", "Storing Data"],
        "current_step": 0,
        "status": "In Progress"
    })
    try:
        # Parse Email Content and Generate Prompt
        prompt = parse_email_content(email)
        await progress_tracker.update_step(task_id, 1)

        # Call Gemini API with Prompt
        parsed_data = await call_gemini_api(prompt)
        await progress_tracker.update_step(task_id, 2)

        # Insert Parsed Data into BigQuery
        await insert_raw_parsed_output(email_id, parsed_data, bigquery_client)
        await progress_tracker.update_step(task_id, 3)

        # Update Task Status to Completed
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Email parsing completed for email ID: {email_id}")

        # Append Result
        results.append({"email_id": email_id, "parsed_data": parsed_data, "status": "Success"})
    except (ParseError, ServerError, InvalidInputError) as e:
        # Update Task Status to Failed
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing email ID {email_id}: {e}")
        # Append Error Result
        results.append({
            "email_id": email_id,
            "error": create_error_response(
                ERROR_CODES['server_error'],
                f"Failed to parse email ID {email_id} due to a server error."
            )
        })
    finally:
        # Remove Task from Progress Tracker
        await progress_tracker.remove_progress(task_id)

async def insert_raw_parsed_output(
    email_id: str,
    raw_output: str,
    bigquery_client: bigquery.Client,
    parser_version: str = "1.0"
):
    table_id = f"{settings.BIGQUERY_PROJECT_ID}.{settings.BIGQUERY_DATASET}.raw_parsed_output"
    rows_to_insert = [{
        "email_id": email_id,
        "parsed_timestamp": datetime.utcnow().isoformat(),
        "raw_parsed_output": raw_output,
        "parser_version": parser_version
    }]
    try:
        await asyncio.to_thread(bigquery_client.insert_rows_json, table_id, rows_to_insert)
        logger.info(
            f"Inserted parsed data into BigQuery for email ID: {email_id}",
            extra={"table_id": table_id, "email_id": email_id}
        )
    except Exception as e:
        logger.error(
            f"Failed to insert raw parsed output for email ID {email_id}: {str(e)}. Data: {rows_to_insert}",
            exc_info=True
        )
        raise ServerError(f"Failed to insert parsed data for email ID {email_id}. {e}")

# ------------------------- Frontend Serving ----------------------------

# Mount the /static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    frontend_path = os.path.join("static", "index.html")
    if not os.path.exists(frontend_path):
        logger.error("Frontend index.html not found.")
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(frontend_path)
