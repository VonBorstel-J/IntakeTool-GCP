# gmail_utils.py
import base64, os, logging, asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache
from ratelimit import limits, sleep_and_retry
from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, before_log, after_log
from fastapi import HTTPException
from google.cloud import documentai_v1 as documentai, storage

logger = logging.getLogger("gmail_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'))
    logger.addHandler(handler)

ERROR_CODES = {"fetch_error": "ERR001", "parse_error": "ERR002", "invalid_input": "ERR003", "server_error": "ERR004"}
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
ONE_MINUTE, MAX_REQUESTS_PER_MINUTE = 60, 100

def get_logger(extra: Dict[str, Any] = {}) -> logging.Logger:
    return logging.LoggerAdapter(logger, extra)

@sleep_and_retry
@limits(calls=MAX_REQUESTS_PER_MINUTE, period=ONE_MINUTE)
def rate_limited_api_call(service_method, *args, **kwargs):
    logger_adapter = get_logger()
    try:
        return service_method(*args, **kwargs).execute()
    except Exception as e:
        logger_adapter.error(f"Rate limited API call failed: {e}")
        raise

@retry(
    retry=retry_if_exception_type((HttpError, Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
def resilient_api_call(service_method, *args, **kwargs):
    return rate_limited_api_call(service_method, *args, **kwargs)

def validate_service_account(credentials):
    if "https://www.googleapis.com/auth/gmail.readonly" not in credentials.scopes:
        raise ValueError("Service account is missing the required Gmail scope.")
    get_logger().info("Service account validated successfully.")

def get_gmail_service() -> Any:
    logger_adapter = get_logger()
    try:
        service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
        if service_account_path and os.path.exists(service_account_path):
            credentials = service_account.Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
            validate_service_account(credentials)
            logger_adapter.info("Using service account credentials.")
        else:
            credentials, _ = default(scopes=SCOPES)
            logger_adapter.info("Using default credentials.")
        return build('gmail', 'v1', credentials=credentials)
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: Failed to create Gmail service. Cause: {e}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to initialize Gmail service."}})

def decode_body(data: str) -> str:
    logger_adapter = get_logger()
    try:
        return base64.urlsafe_b64decode(data).decode('utf-8')
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['parse_error']}: Failed to decode email body. Cause: {e}")
        return ""

def get_attachment_metadata(part: Dict[str, Any]) -> Dict[str, Any]:
    logger_adapter = get_logger()
    try:
        metadata = {
            "filename": part.get('filename') or "UNKNOWN_FILENAME",
            "mimeType": part.get('mimeType') or "application/octet-stream",
            "size": int(part.get('body', {}).get('size', 0)),
            "attachmentId": part.get('body', {}).get('attachmentId'),
            "hash": part.get('body', {}).get('hash')
        }
        logger_adapter.info(f"Extracted metadata for attachment: {metadata['filename']}")
        return metadata
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['parse_error']}: Failed to extract attachment metadata: {e}")
        return {"filename": "UNKNOWN_FILENAME", "mimeType": None, "size": None}

def process_attachment_with_documentai(processor_id, bucket_name, file_name, mime_type):
    try:
        client = documentai.DocumentProcessorServiceClient()
        gcs_input_uri = f"gs://{bucket_name}/{file_name}"
        input_config = documentai.types.RawDocument(content=gcs_input_uri, mime_type=mime_type)
        request = {
            "name": f"projects/{os.getenv('GCP_PROJECT_ID')}/locations/{os.getenv('GCP_LOCATION')}/processors/{processor_id}",
            "raw_document": input_config,
        }
        response = client.process_document(request=request)
        return response.document.text
    except Exception as e:
        logger.error(f"Document AI processing failed for file '{file_name}': {e}")
        return f"Failed to process '{file_name}' due to {e}"

def validate_and_process_attachments(attachments, bucket_name, processor_id):
    processed_texts = []
    for attachment in attachments:
        try:
            validate_attachment(attachment["filename"], attachment["size"])
            gcs_path = upload_attachment_to_gcs(bucket_name, attachment["filename"], attachment["content"])
            attachment_text = process_attachment_with_documentai(
                processor_id, bucket_name, attachment["filename"], attachment.get("mimeType", "application/pdf")
            )
            processed_texts.append(attachment_text)
        except Exception as e:
            logger.error(f"Error handling attachment '{attachment['filename']}': {e}")
            processed_texts.append(f"Error processing '{attachment['filename']}': {e}")
    return " ".join(processed_texts)

def upload_attachment_to_gcs(bucket_name, file_name, file_content):
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(file_name)
    blob.upload_from_string(file_content)
    gcs_uri = f"gs://{bucket_name}/{file_name}"
    if not gcs_uri:
        raise ValueError(f"Failed to generate GCS URI for {file_name}")
    return gcs_uri

def handle_attachments_and_parse(email, attachments, bucket_name, processor_id):
    attachments_content = []
    for attachment in attachments:
        gcs_path = upload_attachment_to_gcs(bucket_name, attachment["filename"], attachment["content"])
        attachment_text = process_attachment_with_documentai(processor_id, bucket_name, attachment["filename"])
        attachments_content.append(attachment_text)
    combined_attachments_content = " ".join(attachments_content)
    context = {
        "email_thread_content": email.get("body", ""),
        "attachments_content": combined_attachments_content,
    }
    return context

def validate_attachment(file_name, file_size):
    allowed_types = {"jpeg", "jpg", "png", "bmp", "pdf", "tiff", "tif", "gif"}
    max_size_mb = 20
    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File size exceeds {max_size_mb}MB limit.")
    if not any(file_name.lower().endswith(ext) for ext in allowed_types):
        raise ValueError(f"Unsupported file type: {file_name}")


def extract_email_metadata(headers: List[Dict[str, str]]) -> Dict[str, str]:
    mapping = {'From': 'sender', 'To': 'recipients', 'Cc': 'cc', 'Bcc': 'bcc', 'Date': 'date', 'Message-ID': 'message_id', 'In-Reply-To': 'in_reply_to', 'References': 'references'}
    return {mapping[h['name']]: h.get('value', '') for h in headers if h.get('name') in mapping}

def extract_message_details(service: Any, message: Dict[str, Any]) -> Dict[str, Any]:
    logger_adapter = get_logger({'message_id': message.get('id')})
    try:
        payload = message.get('payload', {})
        if not payload:
            logger_adapter.warning(f"{ERROR_CODES['parse_error']}: Message {message.get('id')} has no payload.")
            return {}
        headers = payload.get('headers', [])
        metadata = extract_email_metadata(headers)
        subject = next((h['value'] for h in headers if h.get('name') == 'Subject' and h.get('value').strip()), '')
        body = decode_body(payload.get('body', {}).get('data', ''))
        attachments = [decode_attachment(service, message['id'], part.get('body', {}).get('attachmentId'))
                       for part in payload.get('parts', []) if part.get('filename') and part.get('body', {}).get('attachmentId')]
        from_header = next((h['value'] for h in headers if h.get('name') == 'From' and h.get('value').strip()), '')
        timestamp = message.get('internalDate')
        timestamp = datetime.utcfromtimestamp(int(timestamp)/1000).isoformat() + 'Z' if timestamp else ""
        return {
            'message_id': message.get('id'),
            'metadata': metadata,
            'labels': message.get('labelIds', []),
            'from': from_header,
            'timestamp': timestamp,
            'subject': subject,
            'body': body,
            'attachments': [att for att in attachments if att.get('data')]
        }
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Error extracting details from message {message.get('id')}. Cause: {e}")
        return {}

def combine_thread_messages(thread_id: str, processed_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger_adapter = get_logger({'thread_id': thread_id})
    try:
        body = "\n\n".join(msg.get('body', '') for msg in processed_messages).strip()
        attachments = [att for msg in processed_messages for att in msg.get('attachments', [])]
        subject = next((msg['subject'] for msg in processed_messages if msg.get('subject')), "")
        logger_adapter.info(f"Thread {thread_id} contains {len(processed_messages)} messages")
        return {"thread_id": thread_id, "subject": subject, "body": body, "attachments": attachments, "messages": processed_messages}
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Error combining thread messages for thread {thread_id}. Cause: {e}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Failed to combine thread messages."}})

@lru_cache(maxsize=1000)
def fetch_cached_email(email_id: str) -> Dict[str, Any]:
    return asyncio.run(fetch_email(email_id))

def clear_email_cache():
    fetch_cached_email.cache_clear()
    get_logger().info("Email cache cleared")

def reset_cache_stats():
    fetch_cached_email.cache_clear()
    get_logger().info("Cache statistics reset")

def get_cache_info(reset: bool = False) -> Dict[str, int]:
    if reset:
        reset_cache_stats()
    info = fetch_cached_email.cache_info()
    return {"hits": info.hits, "misses": info.misses, "maxsize": info.maxsize, "currsize": info.currsize}

async def check_gmail_api_health() -> Dict[str, Any]:
    logger_adapter = get_logger()
    try:
        service = get_gmail_service()
        result = await asyncio.to_thread(resilient_api_call, service.users().getProfile, 'me')
        return {"status": "healthy", "email": result.get('emailAddress'), "quota": {"limit": MAX_REQUESTS_PER_MINUTE, "period": ONE_MINUTE}}
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

async def get_email_labels(email_id: str) -> List[str]:
    logger_adapter = get_logger({'email_id': email_id})
    service = get_gmail_service()
    try:
        message = await asyncio.to_thread(resilient_api_call, service.users().messages().get, 'me', email_id, {'format': 'minimal'})
        return message.get('labelIds', [])
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch labels for email {email_id}: {e}")
        return []

async def fetch_emails_by_label(label: str, max_results: int = 100) -> List[Dict[str, Any]]:
    logger_adapter = get_logger({'label': label})
    service = get_gmail_service()
    try:
        messages = await asyncio.to_thread(resilient_api_call, service.users().messages().list, 'me', labelIds=[label], maxResults=max_results)
        if not messages.get('messages'):
            logger_adapter.info(f"No messages found with label: {label}")
            return []
        return await fetch_email_batch([msg['id'] for msg in messages.get('messages', [])])
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: Failed to fetch emails with label {label}: {e}")
        return []

def get_batch_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successful = sum(1 for r in results if "error" not in r)
    failed = total - successful
    success_rate = f"{(successful / total * 100):.2f}%" if total else "0.00%"
    return {"total": total, "successful": successful, "failed": failed, "success_rate": success_rate}

def generate_thread_summary(thread: Dict[str, Any]) -> Dict[str, Any]:
    logger_adapter = get_logger({'thread_id': thread.get("thread_id")})
    try:
        timestamps = [msg["timestamp"] for msg in thread["messages"] if msg.get("timestamp")]
        date_range = {"start": min(timestamps) if timestamps else None, "end": max(timestamps) if timestamps else None}
        participants = list({msg["from"] for msg in thread["messages"] if msg.get("from")})
        return {"thread_id": thread["thread_id"], "message_count": len(thread["messages"]), "date_range": date_range, "participants": participants, "subject": thread["subject"]}
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['parse_error']}: Failed to generate thread summary: {e}")
        return {"thread_id": thread.get("thread_id", "unknown"), "error": str(e)}

async def search_emails(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    logger_adapter = get_logger({'query': query})
    if not query or not isinstance(query, str) or not query.strip():
        logger_adapter.error(f"{ERROR_CODES['invalid_input']}: Invalid search query provided.")
        raise HTTPException(status_code=400, detail={"error": {"code": ERROR_CODES['invalid_input'], "message": "Search query cannot be empty or invalid."}})
    service = get_gmail_service()
    try:
        messages = await asyncio.to_thread(resilient_api_call, service.users().messages().list, 'me', q=query, maxResults=max_results)
        message_ids = [msg['id'] for msg in messages.get('messages', [])]
        return await fetch_email_batch(message_ids)
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: Failed to search emails: {e}")
        return []

def decode_attachment(service: Any, message_id: str, attachment_id: str) -> Dict[str, Any]:
    logger_adapter = get_logger({'message_id': message_id, 'attachment_id': attachment_id})
    try:
        attachment = resilient_api_call(service.users().messages().attachments().get, userId='me', messageId=message_id, id=attachment_id)
        if 'data' not in attachment:
            logger_adapter.warning(f"{ERROR_CODES['parse_error']}: Missing 'data' in attachment {attachment_id} for message {message_id}.")
            return {'filename': 'UNKNOWN_FILENAME', 'mimeType': None, 'size': None, 'data': None, 'error': 'Attachment decoding failed'}
        data = base64.urlsafe_b64decode(attachment['data'])
        metadata = {"filename": "UNKNOWN_FILENAME", "mimeType": attachment.get("mimeType", "application/octet-stream"), "size": len(data)}
        logger_adapter.info(f"Decoded attachment: {metadata['filename']}, size: {metadata['size']} bytes")
        return {**metadata, 'data': data}
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Error decoding attachment {attachment_id} for message {message_id}. Cause: {e}")
        return {'filename': 'UNKNOWN_FILENAME', 'mimeType': None, 'size': None, 'data': None, 'error': str(e)}

async def fetch_email(email_id: str) -> Dict[str, Any]:
    logger_adapter = get_logger({'email_id': email_id})
    service = get_gmail_service()
    try:
        logger_adapter.info(f"Fetching email with ID: {email_id}")
        message = await asyncio.to_thread(resilient_api_call, service.users().messages().get, 'me', email_id, {'format': 'full'})
        thread_id = message.get('threadId')
        if not thread_id:
            error_msg = f"{ERROR_CODES['fetch_error']}: Thread ID not found for email ID: {email_id}"
            logger_adapter.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": error_msg}})
        logger_adapter.info(f"Email ID: {email_id} belongs to thread ID: {thread_id}")
    except HttpError as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: HTTP error while fetching email {email_id}. Cause: {e}")
        raise HTTPException(status_code=e.resp.status, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to fetch email due to an HTTP error."}})
    except HTTPException:
        raise
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Unexpected error while fetching email {email_id}. Cause: {e}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error while fetching email."}})
    try:
        logger_adapter.info(f"Fetching thread with ID: {thread_id}")
        thread = await asyncio.to_thread(resilient_api_call, service.users().threads().get, 'me', thread_id, {'format': 'full'})
        messages = thread.get('messages', [])
        if not messages:
            error_msg = f"{ERROR_CODES['fetch_error']}: No messages found in thread ID: {thread_id}"
            logger_adapter.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": error_msg}})
        sorted_messages = sorted(messages, key=lambda msg: int(msg.get('internalDate', '0')))
        logger_adapter.info(f"Fetched {len(sorted_messages)} messages in thread ID: {thread_id}")
    except HttpError as e:
        logger_adapter.error(f"{ERROR_CODES['fetch_error']}: HTTP error while fetching thread {thread_id}. Cause: {e}")
        raise HTTPException(status_code=e.resp.status, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to fetch thread due to an HTTP error."}})
    except HTTPException:
        raise
    except Exception as e:
        logger_adapter.error(f"{ERROR_CODES['server_error']}: Unexpected error while fetching thread {thread_id}. Cause: {e}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error while fetching thread."}})
    processed_messages = [extract_message_details(service, msg) for msg in sorted_messages if extract_message_details(service, msg)]
    if not processed_messages:
        error_msg = f"{ERROR_CODES['parse_error']}: All messages in thread ID: {thread_id} are malformed or empty."
        logger_adapter.error(error_msg)
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['parse_error'], "message": error_msg}})
    return combine_thread_messages(thread_id, processed_messages)

async def fetch_email_batch(email_ids: List[str], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
    batch_size = batch_size or min(10, MAX_REQUESTS_PER_MINUTE)
    results = []
    for i in range(0, len(email_ids), batch_size):
        batch = email_ids[i:i + batch_size]
        batch_results = await asyncio.gather(*[asyncio.to_thread(fetch_cached_email, eid) for eid in batch], return_exceptions=True)
        for res in batch_results:
            if isinstance(res, Exception):
                if isinstance(res, HTTPException):
                    logger.error(f"{res.detail['error']['code']}: {res.detail['error']['message']}")
                    results.append({"error": res.detail["error"]})
                else:
                    logger.error(f"{ERROR_CODES['server_error']}: Error fetching email in batch. Cause: {res}")
                    results.append({"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error during batch email fetching."}})
            else:
                results.append(res)
    return results
