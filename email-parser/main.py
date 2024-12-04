# main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
    get_cache_info,
    insert_raw_parsed_output,
    insert_emails_table,
    insert_assignments_table
)
from gemini_utils import (
    parse_email_content,
    call_gemini_api
)
from google.cloud import bigquery
from datetime import datetime, timedelta, date
import asyncio
from exceptions import FetchError, ParseError, ServerError, InvalidInputError, GCPError
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from config import settings, get_logger  # Ensure config.py is correctly implemented
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import secrets
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google.auth.transport.requests import Request

# -------------------------- Logging Setup ----------------------------
logger = get_logger()

# ----------------------------- FastAPI App ----------------------------
app = FastAPI(
    title="Email Processing Service",
    description="A FastAPI service for processing emails using Gmail and Gemini APIs.",
    version="1.0.0"
)

# ----------------------- Rate Limiting Middleware ---------------------
rate_limit = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[rate_limit]
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=create_error_response(
            ERROR_CODES['rate_limit'],
            "Rate limit exceeded. Please try again later."
        )
    )

# --------------------- Security Headers Middleware ------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Security headers to protect against common vulnerabilities
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

# Ensure SecurityHeadersMiddleware is added first
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
                content=create_error_response(
                    str(he.status_code),
                    he.detail
                )
            )
        except (FetchError, ParseError, ServerError, InvalidInputError, GCPError) as custom_exc:
            # Map exception types to error codes
            exception_mapping = {
                FetchError: ERROR_CODES['fetch_error'],
                ParseError: ERROR_CODES['parse_error'],
                ServerError: ERROR_CODES['server_error'],
                InvalidInputError: ERROR_CODES['invalid_input'],
                GCPError: ERROR_CODES['gcp_error'],
            }
            error_code = exception_mapping.get(type(custom_exc), ERROR_CODES['internal_error'])
            logger.error(f"{error_code}: {str(custom_exc)}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    error_code,
                    str(custom_exc)
                )
            )
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    ERROR_CODES['internal_error'],
                    "An internal server error occurred. Please try again later."
                )
            )

# Add ExceptionHandlingMiddleware after SecurityHeadersMiddleware
app.add_middleware(ExceptionHandlingMiddleware)

# ----------------------- JWT Configuration ----------------------------
SECRET_KEY = settings.SECRET_KEY.get_secret_value() if settings.SECRET_KEY else "default_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# -------------------------- OAuth2 State Management -------------------
class OAuthStateStore:
    """
    A simple in-memory store for OAuth2 state parameters.
    For production, consider using a persistent storage like Redis.
    """
    def __init__(self):
        self._states: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._expiration_seconds = 600  # States expire after 10 minutes

    async def store_state(self, state: str):
        async with self._lock:
            self._states[state] = datetime.utcnow().timestamp() + self._expiration_seconds
            logger.debug(f"Stored OAuth2 state: {state}")

    async def validate_state(self, state: str) -> bool:
        async with self._lock:
            expiry = self._states.get(state)
            if expiry and expiry > datetime.utcnow().timestamp():
                # Valid state
                del self._states[state]
                logger.debug(f"Validated and removed OAuth2 state: {state}")
                return True
            logger.warning(f"Invalid or expired OAuth2 state: {state}")
            return False

    async def cleanup_states(self):
        """
        Cleans up expired states.
        """
        async with self._lock:
            current_time = datetime.utcnow().timestamp()
            expired_states = [state for state, expiry in self._states.items() if expiry < current_time]
            for state in expired_states:
                del self._states[state]
                logger.debug(f"Cleaned up expired OAuth2 state: {state}")

# Initialize global OAuth state store
oauth_state_store = OAuthStateStore()

# Periodically clean up expired states
async def periodic_cleanup():
    while True:
        await asyncio.sleep(600)  # Run cleanup every 10 minutes
        await oauth_state_store.cleanup_states()

@app.on_event("startup")
async def startup_event():
    logger.info("Application is starting up.")
    try:
        app.state.bigquery_client = await get_bigquery_client()
        logger.info("BigQuery client initialized.")
        app.state.oauth_flow = Flow.from_client_secrets_file(
            settings.GOOGLE_CLIENT_SECRETS_FILE,
            scopes=settings.GOOGLE_SCOPES,
            redirect_uri=settings.REDIRECT_URI
        )
        logger.info("OAuth2 flow initialized.")
        # Start periodic cleanup task
        asyncio.create_task(periodic_cleanup())
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down.")
    try:
        if hasattr(app.state, "bigquery_client"):
            await asyncio.to_thread(app.state.bigquery_client.close)
            logger.info("BigQuery client closed.")
    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)

# -------------------------- OAuth2 Configuration ------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------------- Pydantic Models ------------------------------
class ParseEmailRequest(BaseModel):
    email_id: str = Field(..., description="The ID of the email to parse.")

class ParseTextRequest(BaseModel):
    custom_text: str = Field(..., description="The custom text to parse.")

class ProcessBatchRequest(BaseModel):
    email_ids: List[str] = Field(..., description="List of email IDs to process in batch.")

class ParseLabelRequest(BaseModel):
    label_name: str = Field(..., description="The Gmail label name to parse emails from.")

# ------------------------ Authentication Utilities ------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Extracts and returns the current user from the JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("JWT token does not contain 'sub' claim.")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials.")
        return user_id
    except JWTError:
        logger.warning("JWT token decoding failed.")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials.")

# ----------------------- BigQuery Client Dependency -----------------------
async def get_bigquery_client() -> bigquery.Client:
    """
    Provides a BigQuery client instance.
    """
    if not hasattr(app.state, "bigquery_client"):
        app.state.bigquery_client = bigquery.Client(project=settings.BIGQUERY_PROJECT_ID)
    return app.state.bigquery_client

# ------------------------ Error Response Utility ------------------------
TRIAL_GUIDE_URL = "https://example.com/troubleshooting"
ERROR_CODES = {
    "fetch_error": "ERR001",
    "parse_error": "ERR002",
    "invalid_input": "ERR003",
    "server_error": "ERR004",
    "internal_error": "ERR_INTERNAL",
    "gcp_error": "ERR005",
    "rate_limit": "ERR_RATE_LIMIT"
}

def create_error_response(code: str, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "troubleshooting": TRIAL_GUIDE_URL
        }
    }

# ----------------------------- API Endpoints ----------------------------
@app.get("/api/status")
async def read_status():
    logger.info("Status check requested.")
    return {"status": "Service is up and running"}

@app.post("/api/parse-email")
@limiter.limit("5/minute")
async def parse_email_endpoint(
    request: ParseEmailRequest,
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
        await insert_emails_table(email_content, bigquery_client)
        await insert_assignments_table(
            claim_number=parsed_data.get("claim_number"),
            date_of_loss=parsed_data.get("date_of_loss"),
            policy_number=parsed_data.get("policy_number"),
            insured_name=parsed_data.get("insured_name"),
            adjuster=parsed_data.get("adjuster"),
            email_id=email_id,
            bigquery_client=bigquery_client
        )
        await progress_tracker.update_step(task_id, 3)

        # Update Task Status to Completed
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Email parsing completed for email ID: {email_id}")

        return {"email_id": email_id, "parsed_data": parsed_data, "status": "Success"}
    except (FetchError, ParseError, ServerError, InvalidInputError, GCPError) as e:
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing email ID {email_id}: {e}")
        raise e
    finally:
        await progress_tracker.remove_progress(task_id)

@app.post("/api/parse-text")
@limiter.limit("5/minute")
async def parse_text_endpoint(request: ParseTextRequest):
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
    except (ParseError, ServerError, InvalidInputError, GCPError) as e:
        await progress_tracker.set_status(task_id, "Failed")
        logger.error(f"Error parsing custom text: {e}")
        raise e
    finally:
        await progress_tracker.remove_progress(task_id)

@app.post("/api/process-batch")
@limiter.limit("2/minute")
async def process_batch_endpoint(
    request: ProcessBatchRequest,
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
    except GCPError as ge:
        logger.error(f"{ERROR_CODES['gcp_error']}: GCP related error during batch fetch. Cause: {ge}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ERROR_CODES['gcp_error'],
                "GCP related error occurred while fetching emails. Please try again later."
            )
        )
    
    results = []
    tasks = []
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
    request: ParseLabelRequest,
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

        # Fetch Batch Emails
        emails = await fetch_email_batch(email_ids, creds)
        await progress_tracker.update_step(task_id, 2)

        # Process Each Email Individually
        results = []
        tasks = []
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
        
        await progress_tracker.update_step(task_id, 3)
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Label parsing completed for label: {label_name}")
        return {"label_name": label_name, "processed_emails": len(email_ids), "status": "Success", "results": results}
    except (FetchError, ParseError, ServerError, InvalidInputError, GCPError) as e:
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
        # Process Email Content
        email_content = await process_emails_by_label(email_id, email, creds)
        await progress_tracker.update_step(task_id, 1)

        # Parse Email Content and Generate Prompt
        prompt = parse_email_content(email_content)

        # Call Gemini API with Prompt
        parsed_data = await call_gemini_api(prompt)
        await progress_tracker.update_step(task_id, 2)

        # Insert Parsed Data into BigQuery
        await insert_raw_parsed_output(email_id, parsed_data, bigquery_client)
        await insert_emails_table(email_content, bigquery_client)
        await insert_assignments_table(
            claim_number=parsed_data.get("claim_number"),
            date_of_loss=parsed_data.get("date_of_loss"),
            policy_number=parsed_data.get("policy_number"),
            insured_name=parsed_data.get("insured_name"),
            adjuster=parsed_data.get("adjuster"),
            email_id=email_id,
            bigquery_client=bigquery_client
        )
        await progress_tracker.update_step(task_id, 3)

        # Update Task Status to Completed
        await progress_tracker.set_status(task_id, "Completed")
        logger.info(f"Email parsing completed for email ID: {email_id}")

        # Append Result
        results.append({"email_id": email_id, "parsed_data": parsed_data, "status": "Success"})
    except (ParseError, ServerError, InvalidInputError, GCPError) as e:
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

async def get_gmail_credentials(user_id: str = Depends(get_current_user)) -> Credentials:
    """
    Retrieves Gmail credentials for the authenticated user.
    In a stateless setup, credentials might need to be fetched from a secure storage using the user_id.
    """
    # Implement credential retrieval logic based on user_id
    # For example, fetch from a database or secret manager
    credentials_data = await get_credentials_for_user(user_id)
    if not credentials_data:
        logger.warning("User not authorized. Credentials not found.")
        raise HTTPException(status_code=401, detail="User not authorized. Please log in.")

    creds = Credentials(
        token=credentials_data['token'],
        refresh_token=credentials_data.get('refresh_token'),
        token_uri=credentials_data['token_uri'],
        client_id=credentials_data['client_id'],
        client_secret=credentials_data['client_secret'],
        scopes=credentials_data['scopes']
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Update credentials in secure storage
            await update_credentials_for_user(user_id, creds)
            logger.info("Credentials refreshed successfully.")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}", exc_info=True)
            raise HTTPException(status_code=401, detail="Token refresh failed. Please log in again.")

    return creds

async def get_credentials_for_user(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves stored credentials for a given user_id.
    Replace this with actual logic to fetch credentials from your storage solution.
    """
    # TODO: Implement credential retrieval logic
    # Example using an in-memory store (not suitable for production)
    # Replace with database queries or secure storage access
    return app.state.credentials_store.get(user_id) if hasattr(app.state, 'credentials_store') else None

async def update_credentials_for_user(user_id: str, creds: Credentials):
    """
    Updates stored credentials for a given user_id.
    Replace this with actual logic to update credentials in your storage solution.
    """
    # TODO: Implement credential update logic
    # Example using an in-memory store (not suitable for production)
    # Replace with database updates or secure storage access
    if not hasattr(app.state, 'credentials_store'):
        app.state.credentials_store = {}
    app.state.credentials_store[user_id] = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

# --------------------------- Progress Tracker --------------------------
class ProgressTracker:
    def __init__(self):
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def set_progress(self, task_id: str, progress: Dict[str, Any]):
        async with self._lock:
            self._progress[task_id] = progress
            logger.debug(f"Set progress for task {task_id}: {progress}")
    
    async def update_step(self, task_id: str, step: int):
        async with self._lock:
            if task_id in self._progress:
                self._progress[task_id]["current_step"] = step
                logger.debug(f"Updated step {step} for task {task_id}")
    
    async def set_status(self, task_id: str, status: str):
        async with self._lock:
            if task_id in self._progress:
                self._progress[task_id]["status"] = status
                logger.debug(f"Set status '{status}' for task {task_id}")
    
    async def get_progress(self, task_id: str):
        async with self._lock:
            return self._progress.get(task_id)
    
    async def remove_progress(self, task_id: str):
        async with self._lock:
            if task_id in self._progress:
                del self._progress[task_id]
                logger.debug(f"Removed progress for task {task_id}")

# Initialize global progress tracker
progress_tracker = ProgressTracker()

# ----------------------------- Path Handling ------------------------------
# Serve static files using absolute paths
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    frontend_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(frontend_path):
        logger.error("Frontend index.html not found.")
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(frontend_path)
