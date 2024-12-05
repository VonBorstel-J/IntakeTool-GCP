# gmail_utils.py

import asyncio
import base64
import hashlib
import os
import time
from typing import Any, Dict, List, Optional, TypedDict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import documentai_v1 as documentai
from google.cloud import storage, vision, bigquery  # BigQuery import

from config import settings, get_logger
from exceptions import FetchError, ParseError, ServerError, GCPError, InvalidInputError

from datetime import datetime

logger = get_logger()

# ------------------------- Environment Variable Validation -------------------

def validate_env_variables():
    """
    Validates critical environment variables at startup.
    Logs warnings for missing or malformed variables without crashing.
    """
    required_vars = {
        "GMAIL_API_KEY": settings.GMAIL_API_KEY,
        "EMAIL_ATTACHMENT_BUCKET": settings.EMAIL_ATTACHMENT_BUCKET,
        "SECRET_KEY": settings.SECRET_KEY,
    }

    for var, value in required_vars.items():
        if not value:
            logger.warning(f"Missing required environment variable: {var}")
        else:
            logger.debug(f"Environment variable {var} is set.")

    # Additional validation can be added here (e.g., format checks)
    # Example: Check if GMAIL_API_KEY has expected format
    if settings.GMAIL_API_KEY and not settings.GMAIL_API_KEY.startswith("AIza"):
        logger.warning("GMAIL_API_KEY format is unexpected.")

# Call the validation function at module load
validate_env_variables()

# ------------------------- Caching Mechanism ----------------------------

class InMemoryCache:
    """
    Simple in-memory cache with expiration.
    For production, replace with Redis or another persistent caching solution.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry['expiry'] > time.time():
                    logger.debug(f"Cache hit for key: {key}")
                    return entry['value']
                else:
                    logger.debug(f"Cache expired for key: {key}")
                    del self._cache[key]
            logger.debug(f"Cache miss for key: {key}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """
        Set a value in the cache with a Time-To-Live (TTL).
        """
        async with self._lock:
            self._cache[key] = {
                'value': value,
                'expiry': time.time() + ttl
            }
            logger.debug(f"Cache set for key: {key} with TTL: {ttl}s")

    async def clear(self):
        """
        Clear the entire cache.
        """
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared.")

    async def get_info(self) -> Dict[str, Any]:
        """
        Retrieve cache information.
        """
        async with self._lock:
            current_time = time.time()
            active_entries = {k: v for k, v in self._cache.items() if v['expiry'] > current_time}
            return {
                "total_entries": len(active_entries),
                "expired_entries": len(self._cache) - len(active_entries),
                "cache_size_bytes": sum(
                    len(k.encode('utf-8')) + len(v['value'].encode('utf-8')) if isinstance(v['value'], str) else 0
                    for k, v in active_entries.items()
                )
            }

# Initialize global cache
email_cache = InMemoryCache()

# ------------------------- GCP Clients Management -----------------------

class GCPClients:
    gmail_service: Optional[Any] = None
    gcs_client: Optional[storage.Client] = None
    vision_client: Optional[vision.ImageAnnotatorClient] = None
    document_ai_client: Optional[documentai.DocumentProcessorServiceClient] = None
    bigquery_client: Optional[bigquery.Client] = None  # BigQuery client
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def initialize_clients(cls, creds: Credentials):
        async with cls._lock:
            if cls.gmail_service is None:
                try:
                    cls.gmail_service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
                    logger.info("Gmail service initialized.", extra={"service": "gmail_service"})
                except Exception as e:
                    logger.error(
                        "Failed to initialize Gmail service.",
                        exc_info=True,
                        extra={"service": "gmail_service", "error": str(e)}
                    )
                    raise GCPError("Failed to initialize Gmail service.")

            if cls.gcs_client is None:
                try:
                    cls.gcs_client = storage.Client()
                    logger.info("GCS client initialized.", extra={"service": "storage.Client"})
                except Exception as e:
                    logger.error(
                        "Failed to initialize GCS client.",
                        exc_info=True,
                        extra={"service": "storage.Client", "error": str(e)}
                    )
                    raise GCPError("Failed to initialize GCS client.")

            if cls.vision_client is None:
                try:
                    cls.vision_client = vision.ImageAnnotatorClient()
                    logger.info("Vision client initialized.", extra={"service": "vision.ImageAnnotatorClient"})
                except Exception as e:
                    logger.error(
                        "Failed to initialize Vision client.",
                        exc_info=True,
                        extra={"service": "vision.ImageAnnotatorClient", "error": str(e)}
                    )
                    raise GCPError("Failed to initialize Vision client.")

            if cls.document_ai_client is None:
                try:
                    cls.document_ai_client = documentai.DocumentProcessorServiceClient()
                    logger.info("Document AI client initialized.", extra={"service": "DocumentProcessorServiceClient"})
                except Exception as e:
                    logger.error(
                        "Failed to initialize Document AI client.",
                        exc_info=True,
                        extra={"service": "DocumentProcessorServiceClient", "error": str(e)}
                    )
                    raise GCPError("Failed to initialize Document AI client.")

            if cls.bigquery_client is None:
                try:
                    cls.bigquery_client = bigquery.Client()
                    logger.info("BigQuery client initialized.", extra={"service": "bigquery.Client"})
                except Exception as e:
                    logger.error(
                        "Failed to initialize BigQuery client.",
                        exc_info=True,
                        extra={"service": "bigquery.Client", "error": str(e)}
                    )
                    raise GCPError("Failed to initialize BigQuery client.")

# ------------------------- Resilient API Call ----------------------------

async def resilient_api_call(func, *args, **kwargs):
    """
    Executes a function in a thread and handles HTTP errors.
    Implements retries for transient errors with exponential backoff.
    """
    max_retries = 5  # Number of retries
    initial_delay = 1  # Initial delay in seconds
    backoff_factor = 2  # Exponential backoff factor

    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except HttpError as e:
            status = e.resp.status
            if status in [429, 500, 502, 503, 504]:
                # Transient errors, retry
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                logger.warning(
                    f"Transient HTTP error {status} encountered. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})",
                    extra={"error_code": status}
                )
                await asyncio.sleep(delay)
                continue
            else:
                # Non-retriable error
                logger.error(
                    "Non-retriable HTTP error during API call.",
                    exc_info=True,
                    extra={"error_code": status}
                )
                raise GCPError(f"HTTP error: {e}")
        except Exception as e:
            # For other exceptions, decide if retry is appropriate
            delay = initial_delay * (backoff_factor ** (attempt - 1))
            logger.warning(
                f"Unexpected error encountered. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})",
                exc_info=True,
                extra={"error": str(e)}
            )
            await asyncio.sleep(delay)
            continue
    # After all retries have been exhausted
    logger.error(
        "Maximum retry attempts reached. Operation failed.",
        extra={"function": func.__name__}
    )
    raise ServerError("Maximum retry attempts reached.")

# ------------------------- Caching Functions -----------------------------

async def get_cache_info() -> Dict[str, Any]:
    """
    Retrieves information about the current state of the email cache.
    """
    info = await email_cache.get_info()
    logger.info("Retrieved cache information.", extra={"cache_info": info})
    return info

async def clear_email_cache():
    """
    Clears the email cache.
    """
    await email_cache.clear()
    logger.info("Email cache cleared successfully.")

# ------------------------- Credential Management -------------------------

# Note: Credential management is handled in main.py. Ensure that gmail_utils.py does not redefine these.

# ------------------------- Attachment Handling ----------------------------

async def decode_attachment(service, message_id: str, attachment_id: str) -> bytes:
    """
    Decodes a Gmail attachment.
    """
    try:
        attachment = await resilient_api_call(
            service.users().messages().attachments().get,
            userId='me',
            messageId=message_id,
            id=attachment_id
        )
        data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
        logger.debug(f"Decoded attachment {attachment_id} for message {message_id}.")
        return data
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['fetch_error']}: Failed to decode attachment.",
            exc_info=True,
            extra={"message_id": message_id, "attachment_id": attachment_id, "error": str(e)}
        )
        raise FetchError("Failed to decode attachment.")

async def upload_to_gcs(file_name: str, file_content: bytes, folder: str = "attachments/") -> str:
    """
    Uploads a file to Google Cloud Storage.
    """
    try:
        client = await get_gcs_client()
        bucket = client.bucket(settings.EMAIL_ATTACHMENT_BUCKET)
        blob = bucket.blob(f"{folder}{file_name}")  # Include folder prefix
        await asyncio.to_thread(blob.upload_from_string, file_content)
        gcs_uri = f"gs://{settings.EMAIL_ATTACHMENT_BUCKET}/{folder}{file_name}"
        logger.info("Uploaded file to GCS.", extra={"gcs_uri": gcs_uri, "file_name": file_name})
        return gcs_uri
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Failed to upload file to GCS.",
            exc_info=True,
            extra={"file_name": file_name, "error": str(e)}
        )
        raise GCPError(f"Failed to upload {file_name} to GCS.")

SUPPORTED_MIME_TYPES = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/png',
    # Add more supported MIME types as needed
]

def validate_mime_type(mime_type: str):
    """
    Validates if the MIME type is supported.
    Logs a warning for unsupported MIME types.
    """
    if mime_type not in SUPPORTED_MIME_TYPES and not mime_type.startswith('image/'):
        logger.warning("Unsupported MIME type detected.", extra={"mime_type": mime_type})
        raise InvalidInputError(f"Unsupported MIME type: {mime_type}")

async def process_pdf(gcs_uri: str, mime_type: str) -> str:
    """
    Processes a PDF using Document AI and extracts text.
    """
    validate_mime_type(mime_type)
    try:
        client = await get_document_ai_client()
        name = f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.ATTACHMENT_OCR_REGION}/processors/{settings.DOCUMENT_AI_PROCESSOR_ID}"
        document = documentai.RawDocument(
            gcs_content_uri=gcs_uri,
            mime_type=mime_type
        )
        request = documentai.ProcessRequest(name=name, raw_document=document)
        response = await asyncio.to_thread(client.process_document, request)
        extracted_text = response.document.text
        logger.info(
            "Processed PDF with Document AI.",
            extra={"gcs_uri": gcs_uri, "extracted_text_length": len(extracted_text)}
        )
        return extracted_text
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Document AI processing failed for {gcs_uri}.",
            exc_info=True,
            extra={"gcs_uri": gcs_uri, "error": str(e)}
        )
        raise ServerError(f"Document AI processing failed for {gcs_uri}.")

async def process_image(gcs_uri: str) -> str:
    """
    Processes an image using Vision AI and extracts text.
    """
    try:
        client = await get_vision_client()
        image = vision.Image(source=vision.ImageSource(gcs_image_uri=gcs_uri))
        response = await asyncio.to_thread(client.text_detection, image)
        texts = response.text_annotations
        extracted_text = texts[0].description if texts else ""
        logger.info(
            "Processed image with Vision AI.",
            extra={"gcs_uri": gcs_uri, "extracted_text_length": len(extracted_text)}
        )
        return extracted_text
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Vision AI processing failed for {gcs_uri}.",
            exc_info=True,
            extra={"gcs_uri": gcs_uri, "error": str(e)}
        )
        raise ServerError(f"Vision AI processing failed for {gcs_uri}.")

async def process_attachment(attachment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes an attachment by uploading to GCS and extracting relevant information.
    """
    filename, mime_type, gcs_uri = attachment['filename'], attachment['mimeType'], attachment.get('gcs_uri')
    if not gcs_uri:
        logger.info("No GCS URI found for attachment.", extra={"filename": filename})
        return None
    try:
        if mime_type in [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]:
            extracted_text = await process_pdf(gcs_uri, mime_type)
        elif mime_type.startswith('image/'):
            extracted_text = await process_image(gcs_uri)
        else:
            # This should already be handled by validate_mime_type, but added as an extra safeguard
            logger.warning(
                "Unsupported MIME type encountered during processing.",
                extra={"mime_type": mime_type, "filename": filename}
            )
            return None
        # Fetch attachment size
        client = await get_gcs_client()
        bucket = client.bucket(settings.EMAIL_ATTACHMENT_BUCKET)
        blob = bucket.blob(f"attachments/{filename}")
        attachment_size = blob.size
        return {
            "attachment_name": filename,
            "attachment_type": mime_type,
            "attachment_size": attachment_size,
            "gcs_uri": gcs_uri
        }
    except InvalidInputError:
        # Already logged in validate_mime_type
        return None
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Failed to process attachment {filename}.",
            exc_info=True,
            extra={"filename": filename, "error": str(e)}
        )
        raise ServerError(f"Failed to process attachment {filename}.")

# ------------------------- Gmail Service Functions ------------------------

async def get_gmail_service(creds: Credentials) -> Any:
    """
    Retrieves the Gmail service client.
    """
    await GCPClients.initialize_clients(creds)
    if GCPClients.gmail_service is None:
        logger.error("Gmail service failed to initialize.", extra={"service": "gmail_service"})
        raise GCPError("Gmail service failed to initialize.")
    return GCPClients.gmail_service

async def get_gcs_client() -> storage.Client:
    """
    Retrieves the GCS client.
    """
    if GCPClients.gcs_client is None:
        logger.error("GCS client not initialized.", extra={"service": "storage.Client"})
        raise GCPError("GCS client not initialized.")
    return GCPClients.gcs_client

async def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    Retrieves the Vision AI client.
    """
    if GCPClients.vision_client is None:
        logger.error("Vision client not initialized.", extra={"service": "vision.ImageAnnotatorClient"})
        raise GCPError("Vision client not initialized.")
    return GCPClients.vision_client

async def get_document_ai_client() -> documentai.DocumentProcessorServiceClient:
    """
    Retrieves the Document AI client.
    """
    if GCPClients.document_ai_client is None:
        logger.error("Document AI client not initialized.", extra={"service": "DocumentProcessorServiceClient"})
        raise GCPError("Document AI client not initialized.")
    return GCPClients.document_ai_client

async def get_bigquery_client() -> bigquery.Client:
    """
    Retrieves the BigQuery client.
    """
    if GCPClients.bigquery_client is None:
        logger.error("BigQuery client not initialized.", extra={"service": "bigquery.Client"})
        raise GCPError("BigQuery client not initialized.")
    return GCPClients.bigquery_client

# ------------------------- Email Processing Functions ---------------------

async def fetch_email(email_id: str, creds: Credentials) -> Dict[str, Any]:
    """
    Fetches and processes a single email by its ID.
    Utilizes caching to prevent redundant API calls.
    """
    cache_key = f"email_{email_id}"
    cached_email = await email_cache.get(cache_key)
    if cached_email:
        logger.info(f"Retrieved email ID {email_id} from cache.")
        return cached_email

    try:
        service = await get_gmail_service(creds)
        message = await resilient_api_call(
            service.users().messages().get,
            userId='me',
            id=email_id,
            format='full'
        )
        thread_id = message.get('threadId')
        if not thread_id:
            error_msg = f"{ERROR_CODES['fetch_error']}: Thread ID not found for email ID: {email_id}"
            logger.error(error_msg, extra={"email_id": email_id})
            raise FetchError(error_msg)
        combined_thread = await fetch_thread(thread_id, creds)
        await email_cache.set(cache_key, combined_thread, ttl=600)  # Cache for 10 minutes
        logger.info(f"Fetched and cached email ID: {email_id}")
        return combined_thread
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['fetch_error']}: Failed to fetch email ID {email_id}.",
            exc_info=True,
            extra={"email_id": email_id, "error": str(e)}
        )
        raise FetchError(f"Failed to fetch email {email_id}.")

async def fetch_email_batch(email_ids: List[str], creds: Credentials) -> List[Dict[str, Any]]:
    """
    Fetches and processes a batch of emails concurrently.
    Returns a list of processed email contents.
    """
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
    results: List[Dict[str, Any]] = []

    async def handle_email(eid: str):
        async with semaphore:
            try:
                email_content = await fetch_email(eid, creds)
                results.append(email_content)
            except FetchError as fe:
                logger.error(f"FetchError for email ID {eid}: {fe}")
                results.append({"message_id": eid, "error": str(fe)})
            except Exception as e:
                logger.error(f"Unexpected error for email ID {eid}: {e}", exc_info=True)
                results.append({"message_id": eid, "error": "Unexpected error during processing."})

    await asyncio.gather(*[handle_email(eid) for eid in email_ids])
    logger.info(f"Batch fetch completed for {len(email_ids)} emails.")
    return results

async def list_emails_by_label(label_name: str, creds: Credentials) -> List[str]:
    """
    Retrieves a list of email IDs that have the specified label.
    """
    try:
        service = await get_gmail_service(creds)
        # Fetch the label ID based on label name
        labels_response = await resilient_api_call(
            service.users().labels().list,
            userId='me'
        )
        labels = labels_response.get('labels', [])
        label_id = next((label['id'] for label in labels if label['name'].lower() == label_name.lower()), None)
        if not label_id:
            error_msg = f"Label '{label_name}' not found."
            logger.error(error_msg, extra={"label_name": label_name})
            raise FetchError(error_msg)
        
        logger.info("Fetched label ID.", extra={"label_name": label_name, "label_id": label_id})
        
        # Fetch messages with the specified label
        messages_response = await resilient_api_call(
            service.users().messages().list,
            userId='me',
            labelIds=[label_id]
        )
        messages = messages_response.get('messages', [])
        email_ids = [msg['id'] for msg in messages]
        
        logger.info("Fetched emails by label.", extra={"label_name": label_name, "email_count": len(email_ids)})
        return email_ids
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['fetch_error']}: Failed to list emails by label '{label_name}'.",
            exc_info=True,
            extra={"label_name": label_name, "error": str(e)}
        )
        raise FetchError(f"Failed to list emails by label '{label_name}'.")

async def process_emails_by_label(label_name: str, creds: Credentials) -> List[Dict[str, Any]]:
    """
    Fetches and processes all emails under a specific Gmail label.
    """
    try:
        email_ids = await list_emails_by_label(label_name, creds)
        if not email_ids:
            logger.info("No emails found for the specified label.", extra={"label_name": label_name})
            return []
        processed_emails = await fetch_email_batch(email_ids, creds)
        return processed_emails
    except Exception as e:
        logger.error(
            f"{ERROR_CODES['fetch_error']}: Failed to process emails by label '{label_name}'.",
            exc_info=True,
            extra={"label_name": label_name, "error": str(e)}
        )
        raise FetchError(f"Failed to process emails by label '{label_name}'.")

# ------------------------- Message Processing -----------------------------

def extract_email_metadata(headers: List[Dict[str, str]], keys: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extracts specified metadata from email headers.
    """
    mapping = {
        'From': 'sender',
        'To': 'recipients',
        'Cc': 'cc',
        'Bcc': 'bcc',
        'Date': 'date',
        'Message-ID': 'message_id',
        'In-Reply-To': 'in_reply_to',
        'References': 'references',
        'Subject': 'subject',
    }
    if keys:
        mapping = {k: v for k, v in mapping.items() if k in keys}
    return {mapping.get(h['name'], h['name']): h.get('value', '') for h in headers if h.get('name') in mapping}

async def extract_message_details(service: Any, message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts details from a Gmail message, including metadata, body, and attachments.
    """
    try:
        payload = message.get('payload', {})
        metadata = extract_email_metadata(payload.get('headers', []))
        body_data = payload.get('body', {}).get('data', '')
        body = base64.urlsafe_b64decode(body_data.encode('UTF-8')).decode('utf-8') if body_data else ""
        attachments = []
        parts = payload.get('parts', []) or []
        for part in parts:
            if part.get('filename') and part.get('body', {}).get('attachmentId'):
                data = await decode_attachment(service, message['id'], part['body']['attachmentId'])
                attachments.append({
                    'filename': part['filename'],
                    'mimeType': part['mimeType'],
                    'data': data,
                })
        return {
            'message_id': message.get('id'),
            'thread_id': message.get('threadId'),
            'metadata': metadata,
            'body': body,
            'attachments': attachments,
        }
    except Exception as e:
        logger.error(
            "Error extracting message details.",
            exc_info=True,
            extra={"message_id": message.get('id'), "error": str(e)}
        )
        raise ParseError("Error extracting message details.")

def combine_thread_messages(thread_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combines messages within a thread, deduplicates attachments, and aggregates the email body.
    """
    try:
        body = "\n\n".join(msg.get('body', '') for msg in messages).strip()
        seen = set()
        deduped_attachments = []
        for msg in messages:
            for att in msg.get('attachments', []):
                key = f"{att['filename']}_{hashlib.md5(att['data']).hexdigest()}"
                if key not in seen:
                    seen.add(key)
                    deduped_attachments.append(att)
        subject = next((msg['metadata'].get('subject') for msg in messages if msg['metadata'].get('subject')), "")
        logger.info(
            "Combined thread messages.",
            extra={"thread_id": thread_id, "message_count": len(messages)}
        )
        return {
            "message_id": messages[-1]['message_id'],  # Latest message ID
            "thread_id": thread_id,
            "metadata": {"subject": subject},
            "body": body,
            "attachments": deduped_attachments
        }
    except Exception as e:
        logger.error(
            "Error combining thread messages.",
            exc_info=True,
            extra={"thread_id": thread_id, "error": str(e)}
        )
        raise ServerError("Failed to combine thread messages.")

async def fetch_thread(thread_id: str, creds: Credentials) -> Dict[str, Any]:
    """
    Fetches all messages in a thread and combines their details.
    """
    try:
        service = await get_gmail_service(creds)
        thread = await resilient_api_call(
            service.users().threads().get,
            userId='me',
            id=thread_id,
            format='full'
        )
        messages = thread.get('messages', [])
        if not messages:
            error_msg = f"{ERROR_CODES['fetch_error']}: No messages found in thread ID: {thread_id}"
            logger.error(error_msg, extra={"thread_id": thread_id})
            raise FetchError(error_msg)
        sorted_messages = sorted(messages, key=lambda msg: int(msg.get('internalDate', '0')))
        tasks = [extract_message_details(service, msg) for msg in sorted_messages]
        messages_details = await asyncio.gather(*tasks)
        combined_thread = combine_thread_messages(thread_id, messages_details)
        return combined_thread
    except HttpError as e:
        logger.error(
            "HTTP error during fetch_thread API call.",
            exc_info=True,
            extra={"error_code": e.resp.status}
        )
        raise GCPError(f"HTTP error: {e}")
    except Exception as e:
        logger.error(
            "Error fetching thread.",
            exc_info=True,
            extra={"thread_id": thread_id, "error": str(e)}
        )
        raise ServerError(f"Error fetching thread {thread_id}.")

# ------------------------- Email Content Processing -----------------------

async def process_email(email_id: str, creds: Credentials) -> Dict[str, Any]:
    """
    Processes a single email by handling attachments and preparing the content.
    """
    try:
        email_content = await fetch_email(email_id, creds)
        attachments = email_content.get('attachments', [])
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)

        async def handle_attachment(att: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                file_name, file_content = att['filename'], att['data']
                # Upload to GCS
                gcs_uri = await upload_to_gcs(file_name, file_content, folder="attachments/")
                att['gcs_uri'] = gcs_uri
                # Process attachment if necessary
                processed_att = await process_attachment(att)
                return processed_att

        tasks = [handle_attachment(att) for att in attachments]
        attachments_content = await asyncio.gather(*tasks)
        attachments_content = [att for att in attachments_content if att]
        processed_email = {
            "message_id": email_content.get("message_id"),
            "thread_id": email_content.get("thread_id"),
            "metadata": email_content.get("metadata", {}),
            "body": email_content.get("body", ""),
            "attachments": attachments_content
        }
        logger.info(f"Processed email ID: {email_id}")
        return processed_email
    except Exception as e:
        logger.error(
            f"Failed to process email ID {email_id}: {e}",
            exc_info=True,
            extra={"email_id": email_id, "error": str(e)}
        )
        raise ParseError(f"Failed to process email ID {email_id}: {e}")

# ------------------------- BigQuery Insertion Functions --------------------

async def insert_raw_parsed_output(
    email_id: str,
    raw_output: str,
    bigquery_client: bigquery.Client,
    parser_version: str = "1.0"
):
    """
    Inserts raw parsed output into BigQuery.
    """
    table_id = f"{settings.BIGQUERY_PROJECT_ID}.{settings.BIGQUERY_DATASET}.raw_parsed_output"
    rows_to_insert = [{
        "email_id": email_id,
        "parsed_timestamp": datetime.utcnow(),
        "raw_parsed_output": raw_output,
        "parser_version": parser_version
    }]
    try:
        errors = await asyncio.to_thread(bigquery_client.insert_rows_json, table_id, rows_to_insert)
        if errors:
            raise ServerError(f"BigQuery insertion errors: {errors}")
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

async def insert_emails_table(
    email_content: Dict[str, Any],
    bigquery_client: bigquery.Client
):
    """
    Inserts email content into BigQuery emails_table.
    """
    table_id = f"{settings.BIGQUERY_PROJECT_ID}.{settings.BIGQUERY_DATASET}.emails_table"
    attachments = email_content.get('attachments', [])
    formatted_attachments = [
        {
            "attachment_name": att.get('attachment_name'),
            "attachment_type": att.get('attachment_type'),
            "attachment_size": att.get('attachment_size'),
            "gcs_uri": att.get('gcs_uri')
        } for att in attachments
    ]
    recipient_emails = email_content.get("metadata", {}).get("recipients", "")
    recipient_emails = [email.strip() for email in recipient_emails.split(",")] if recipient_emails else []

    rows_to_insert = [{
        "email_id": email_content.get("message_id"),
        "thread_id": email_content.get("thread_id"),
        "received_timestamp": datetime.utcnow(),  # Adjust as needed
        "sender_email": email_content.get("metadata", {}).get("sender"),
        "recipient_email": recipient_emails,
        "subject": email_content.get("metadata", {}).get("subject"),
        "email_body": email_content.get("body", ""),
        "attachments": formatted_attachments,
        "full_thread": email_content.get("body", "")  # Adjust as needed
    }]
    try:
        errors = await asyncio.to_thread(bigquery_client.insert_rows_json, table_id, rows_to_insert)
        if errors:
            raise ServerError(f"BigQuery insertion errors: {errors}")
        logger.info(
            f"Inserted data into emails_table for email ID: {email_content.get('message_id')}",
            extra={"table_id": table_id, "email_id": email_content.get("message_id")}
        )
    except Exception as e:
        logger.error(
            f"Failed to insert into emails_table for email ID {email_content.get('message_id')}: {e}",
            exc_info=True
        )
        raise ServerError(f"Failed to insert emails data for email ID {email_content.get('message_id')}. {e}")

async def insert_assignments_table(
    claim_number: str,
    date_of_loss: datetime,
    policy_number: Optional[str],
    insured_name: Optional[str],
    adjuster: Optional[str],
    email_id: Optional[str],
    bigquery_client: bigquery.Client
):
    """
    Inserts assignment data into BigQuery assignments table.
    """
    if not claim_number or not date_of_loss:
        logger.warning("Missing required fields for assignments_table insertion.")
        raise InvalidInputError("Missing required fields: claim_number and date_of_loss.")
    
    table_id = f"{settings.BIGQUERY_PROJECT_ID}.{settings.BIGQUERY_DATASET}.assignments"
    rows_to_insert = [{
        "claim_number": claim_number,
        "date_of_loss": date_of_loss.isoformat(),
        "policy_number": policy_number,
        "insured_name": insured_name,
        "adjuster": adjuster,
        "email_id": email_id
    }]
    try:
        errors = await asyncio.to_thread(bigquery_client.insert_rows_json, table_id, rows_to_insert)
        if errors:
            raise ServerError(f"BigQuery insertion errors: {errors}")
        logger.info(
            f"Inserted data into assignments for claim number: {claim_number}",
            extra={"table_id": table_id, "claim_number": claim_number}
        )
    except Exception as e:
        logger.error(
            f"Failed to insert into assignments for claim number {claim_number}: {e}",
            exc_info=True
        )
        raise ServerError(f"Failed to insert assignments data for claim number {claim_number}. {e}")

# ------------------------- Batch Processing Functions ----------------------

async def process_email_batch(email_ids: List[str], creds: Credentials) -> List[Dict[str, Any]]:
    """
    Processes a batch of emails, ensuring batching logic aligns with prompts.yaml.
    Splits batches if max_tokens limit is exceeded.
    """
    batch_size = settings.BATCH_PROCESSING.get('batch_size', 100)
    max_tokens = settings.AI.get('generative_ai', {}).get('google', {}).get('max_tokens', 250000)
    batches = [email_ids[i:i + batch_size] for i in range(0, len(email_ids), batch_size)]
    processed_results = []

    for batch in batches:
        # Here, you would implement logic to estimate token counts
        # For simplicity, assuming each email contributes a fixed number of tokens
        estimated_tokens = len(batch) * settings.AI.get('generative_ai', {}).get('google', {}).get('max_tokens', 250000) // batch_size
        if estimated_tokens > max_tokens:
            # Further split the batch
            sub_batches = [batch[i:i + batch_size // 2] for i in range(0, len(batch), batch_size // 2)]
            for sub_batch in sub_batches:
                processed_results.extend(await fetch_email_batch(sub_batch, creds))
        else:
            processed_results.extend(await fetch_email_batch(batch, creds))
        logger.info(f"Processed batch of {len(batch)} emails.")
    
    return processed_results

# ------------------------- Data Validation Functions -----------------------

def validate_bigquery_schema(parsed_output: Dict[str, Any], schema: List[Dict[str, Any]]) -> bool:
    """
    Validates the parsed output against the provided BigQuery schema.
    Logs warnings or errors for missing required fields.
    """
    schema_fields = {field['name']: field for field in schema}
    missing_required_fields = [field for field in schema_fields.values() if field['mode'] == 'REQUIRED' and field['name'] not in parsed_output]
    
    if missing_required_fields:
        missing_fields = ", ".join(field['name'] for field in missing_required_fields)
        logger.error(f"Parsed output is missing required fields: {missing_fields}")
        return False
    # Additional type validations can be implemented here
    return True

async def validate_and_insert(parsed_output: Dict[str, Any], bigquery_client: bigquery.Client):
    """
    Validates the parsed output against all BigQuery schemas and inserts them.
    Throws warnings or errors if validation fails.
    """
    # Load schemas
    assignments_schema_path = "assignments_table_schema.json"
    emails_schema_path = "emails_table_schema.json"
    raw_parsed_output_schema_path = "raw_parsed_output_schema.json"

    try:
        with open(assignments_schema_path, 'r') as f:
            assignments_schema = json.load(f)
        with open(emails_schema_path, 'r') as f:
            emails_schema = json.load(f)
        with open(raw_parsed_output_schema_path, 'r') as f:
            raw_parsed_output_schema = json.load(f)
    except Exception as e:
        logger.error("Failed to load BigQuery schemas.", exc_info=True)
        raise ServerError("Failed to load BigQuery schemas.") from e

    # Validate parsed_output against each schema
    if not validate_bigquery_schema(parsed_output, assignments_schema):
        logger.warning("Parsed output does not conform to assignments_table schema.")
    if not validate_bigquery_schema(parsed_output, emails_schema):
        logger.warning("Parsed output does not conform to emails_table schema.")
    if not validate_bigquery_schema(parsed_output, raw_parsed_output_schema):
        logger.warning("Parsed output does not conform to raw_parsed_output schema.")
    
    # Insert into BigQuery tables
    await insert_raw_parsed_output(
        email_id=parsed_output.get("email_id"),
        raw_output=json.dumps(parsed_output),
        bigquery_client=bigquery_client
    )
    await insert_emails_table(
        email_content=parsed_output,
        bigquery_client=bigquery_client
    )
    await insert_assignments_table(
        claim_number=parsed_output.get("claim_number"),
        date_of_loss=parsed_output.get("date_of_loss"),
        policy_number=parsed_output.get("policy_number"),
        insured_name=parsed_output.get("insured_name"),
        adjuster=parsed_output.get("assigner_name"),
        email_id=parsed_output.get("email_id"),
        bigquery_client=bigquery_client
    )

# ------------------------- End of File -------------------------------------
