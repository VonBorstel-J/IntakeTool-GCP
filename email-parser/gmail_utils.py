# gmail_utils.py

import asyncio
import base64
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.cloud import vision
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential
)

from config import get_logger, validate_environment_variables
from exceptions import FetchError, ParseError, ServerError, GCPError

# Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", 10))

ERROR_CODES = {
    "fetch_error": "ERR001",
    "parse_error": "ERR002",
    "invalid_input": "ERR003",
    "server_error": "ERR004",
    "gcp_error": "ERR005"
}

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
DOCUMENT_AI_PROCESSOR_ID = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")

logger = get_logger()

# TypedDicts for better type annotations
class Attachment(TypedDict):
    filename: str
    mimeType: str
    data: bytes
    gcs_uri: Optional[str]

class EmailContent(TypedDict):
    email_content: str
    attachments_content: List[Dict[str, Any]]
    error: Optional[str]

# Retry Configuration
def retry_on_gcp_http():
    return retry_if_exception_type((GCPError, HttpError))

# Gmail Service Initialization
def get_gmail_service() -> Any:
    try:
        validate_environment_variables()
        if GOOGLE_SERVICE_ACCOUNT_PATH and os.path.exists(GOOGLE_SERVICE_ACCOUNT_PATH):
            credentials = service_account.Credentials.from_service_account_file(
                GOOGLE_SERVICE_ACCOUNT_PATH, scopes=SCOPES
            )
            logger.info("Using service account credentials.", extra={"credentials_type": "service_account"})
        else:
            credentials, _ = default(scopes=SCOPES)
            logger.info("Using default credentials.", extra={"credentials_type": "default"})
        return build('gmail', 'v1', credentials=credentials)
    except Exception as e:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to initialize Gmail service.", exc_info=True, extra={"error": str(e)})
        raise FetchError(f"Failed to initialize Gmail service. {e}")

# Resilient API Call with Custom Retry
@retry(
    retry=retry_on_gcp_http(),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def resilient_api_call(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs).execute())
    except (GCPError, HttpError) as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error during API call.", exc_info=True, extra={"error": str(e)})
        raise GCPError(str(e))

# Decode Attachment
async def decode_attachment(service, message_id, attachment_id) -> bytes:
    attachment = await resilient_api_call(
        service.users().messages().attachments().get,
        userId='me',
        messageId=message_id,
        id=attachment_id
    )
    return base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))

# GCS Client
def get_gcs_client() -> storage.Client:
    return storage.Client()

# Vision Client
def get_vision_client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient()

# Upload to GCS with Retry
@retry(
    retry=retry_if_exception_type(GCPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def upload_to_gcs(file_name: str, file_content: bytes) -> str:
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_name)
        await asyncio.get_event_loop().run_in_executor(None, blob.upload_from_string, file_content)
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{file_name}"
        logger.info(f"Uploaded {file_name} to GCS.", extra={"gcs_uri": gcs_uri})
        return gcs_uri
    except Exception as e:
        logger.error(f"{ERROR_CODES['gcp_error']}: Failed to upload {file_name} to GCS.", exc_info=True, extra={"error": str(e)})
        raise GCPError(f"Failed to upload {file_name} to GCS. {e}")

# MIME Type Validation
SUPPORTED_MIME_TYPES = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'image/jpeg', 'image/png']

def validate_mime_type(mime_type: str):
    if mime_type not in SUPPORTED_MIME_TYPES and not mime_type.startswith('image/'):
        raise GCPError(f"Unsupported MIME type: {mime_type}")

# Process PDF with Document AI
@retry(
    retry=retry_if_exception_type(GCPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def process_pdf(gcs_uri: str, mime_type: str) -> str:
    validate_mime_type(mime_type)
    try:
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/processors/{DOCUMENT_AI_PROCESSOR_ID}"
        document = documentai.RawDocument(
            content=await download_from_gcs(gcs_uri),
            mime_type=mime_type
        )
        request = documentai.ProcessRequest(name=name, raw_document=document)
        response = await asyncio.get_event_loop().run_in_executor(None, client.process_document, request)
        extracted_text = response.document.text
        logger.info("Processed PDF with Document AI.", extra={"gcs_uri": gcs_uri, "extracted_text_length": len(extracted_text)})
        return extracted_text
    except Exception as e:
        logger.error(f"{ERROR_CODES['gcp_error']}: Document AI processing failed for {gcs_uri}.", exc_info=True, extra={"error": str(e)})
        raise GCPError(f"Document AI processing failed for {gcs_uri}. {e}")

# Process Image with Vision AI
@retry(
    retry=retry_if_exception_type(GCPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def process_image(gcs_uri: str) -> str:
    try:
        client = get_vision_client()
        image = vision.Image(source=vision.ImageSource(gcs_image_uri=gcs_uri))
        response = await asyncio.get_event_loop().run_in_executor(None, client.text_detection, image)
        texts = response.text_annotations
        extracted_text = texts[0].description if texts else ""
        logger.info("Processed image with Vision AI.", extra={"gcs_uri": gcs_uri, "extracted_text_length": len(extracted_text)})
        return extracted_text
    except Exception as e:
        logger.error(f"{ERROR_CODES['gcp_error']}: Vision AI processing failed for {gcs_uri}.", exc_info=True, extra={"error": str(e)})
        raise GCPError(f"Vision AI processing failed for {gcs_uri}. {e}")

# Download from GCS
async def download_from_gcs(gcs_uri: str) -> bytes:
    try:
        client = get_gcs_client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_name)
        data = await asyncio.get_event_loop().run_in_executor(None, blob.download_as_bytes)
        return data
    except Exception as e:
        logger.error(f"Failed to download from GCS {gcs_uri}.", exc_info=True, extra={"error": str(e)})
        raise GCPError(f"Failed to download from GCS {gcs_uri}. {e}")

# Process Attachment
async def process_attachment(attachment: Attachment) -> Optional[Dict[str, Any]]:
    filename, mime_type, gcs_uri = attachment['filename'], attachment['mimeType'], attachment.get('gcs_uri')
    if not gcs_uri:
        logger.info("No GCS URI found for attachment.", extra={"filename": filename})
        return None
    try:
        if mime_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            extracted_text = await process_pdf(gcs_uri, mime_type)
        elif mime_type.startswith('image/'):
            extracted_text = await process_image(gcs_uri)
        else:
            logger.info("Skipping unsupported attachment type.", extra={"mime_type": mime_type, "filename": filename})
            return None
        return {"name": filename, "extracted_text": extracted_text}
    except Exception as e:
        logger.error(f"{ERROR_CODES['gcp_error']}: Failed to process attachment {filename}.", exc_info=True, extra={"error": str(e)})
        return None

# Extract Email Metadata
def extract_email_metadata(headers: List[Dict[str, str]], keys: Optional[List[str]] = None) -> Dict[str, str]:
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

# Extract Message Details
async def extract_message_details(service: Any, message: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = message.get('payload', {})
        metadata = extract_email_metadata(payload.get('headers', []))
        body_data = payload.get('body', {}).get('data', '')
        body = base64.urlsafe_b64decode(body_data.encode('UTF-8')).decode('utf-8') if body_data else ""
        attachments = []
        for part in payload.get('parts', []):
            if part.get('filename') and part.get('body', {}).get('attachmentId'):
                data = await decode_attachment(service, message['id'], part['body']['attachmentId'])
                attachments.append({
                    'filename': part['filename'],
                    'mimeType': part['mimeType'],
                    'data': data,
                })
        return {
            'message_id': message.get('id'),
            'metadata': metadata,
            'body': body,
            'attachments': attachments,
        }
    except Exception as e:
        logger.error("Error extracting message details.", exc_info=True, extra={"error": str(e)})
        raise ParseError(f"Error extracting message details. {e}")

# Combine Thread Messages with Deduplication
def combine_thread_messages(thread_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        body = "\n\n".join(msg.get('body', '') for msg in messages).strip()
        seen = set()
        deduped_attachments = []
        for msg in messages:
            for att in msg.get('attachments', []):
                key = att['filename']  # Alternatively, use hash(att['data'])
                if key not in seen:
                    seen.add(key)
                    deduped_attachments.append(att)
        subject = next((msg['metadata'].get('subject') for msg in messages if msg['metadata'].get('subject')), "")
        logger.info("Combined thread messages.", extra={"thread_id": thread_id, "message_count": len(messages)})
        return {
            "thread_id": thread_id,
            "subject": subject,
            "body": body,
            "attachments": deduped_attachments,
            "messages": messages,
        }
    except Exception as e:
        logger.error("Error combining thread messages.", exc_info=True, extra={"thread_id": thread_id, "error": str(e)})
        raise ServerError(f"Failed to combine thread messages. {e}")

# Fetch Thread
async def fetch_thread(thread_id: str) -> Dict[str, Any]:
    try:
        service = get_gmail_service()
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
        return combine_thread_messages(thread_id, messages_details)
    except Exception as e:
        logger.error("Error fetching thread.", exc_info=True, extra={"thread_id": thread_id, "error": str(e)})
        raise ServerError(f"Error fetching thread {thread_id}. {e}")

# Fetch Email with Caching
@lru_cache(maxsize=1000)
async def fetch_email(email_id: str) -> Dict[str, Any]:
    try:
        service = get_gmail_service()
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
        return await fetch_thread(thread_id)
    except Exception as e:
        logger.error("Failed to fetch email.", exc_info=True, extra={"email_id": email_id, "error": str(e)})
        raise FetchError(f"Failed to fetch email {email_id}. {e}")

# Process Email
async def process_email(email_id: str) -> Dict[str, Any]:
    email = await fetch_email(email_id)
    attachments = email.get('attachments', [])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def handle_attachment(att: Attachment):
        async with semaphore:
            file_name, file_content = att['filename'], att['data']
            gcs_uri = await upload_to_gcs(file_name, file_content)
            att['gcs_uri'] = gcs_uri
            return await process_attachment(att)

    tasks = [handle_attachment(att) for att in attachments]
    attachments_content = await asyncio.gather(*tasks)
    attachments_content = [att for att in attachments_content if att]
    return {
        "email_content": email.get("body", ""),
        "attachments_content": attachments_content
    }

# Fetch Email Batch with Partial Results
async def fetch_email_batch(email_ids: List[str]) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    results: List[Dict[str, Any]] = []

    async def handle_email(eid: str):
        async with semaphore:
            try:
                email_content = await process_email(eid)
                results.append(email_content)
            except Exception as e:
                logger.error("Error processing email.", exc_info=True, extra={"email_id": eid, "error": str(e)})
                results.append({"error": f"Error processing email {eid}. {e}"})

    await asyncio.gather(*[handle_email(eid) for eid in email_ids])
    return results

# Cache Management
def clear_email_cache():
    fetch_email.cache_clear()
    logger.info("Email cache cleared.")

def reset_cache_stats():
    fetch_email.cache_clear()
    logger.info("Cache statistics reset.")

def get_cache_info(reset: bool = False) -> Dict[str, int]:
    if reset:
        reset_cache_stats()
    info = fetch_email.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize
    }
