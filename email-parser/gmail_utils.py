# gmail_utils.py

import asyncio
import base64
from typing import Any, Dict, List, Optional, TypedDict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.cloud import vision

from config import settings, get_logger
from exceptions import FetchError, ParseError, ServerError, GCPError

from datetime import datetime

logger = get_logger()

# Error Codes (Ensure these match those in config.py or adjust accordingly)
ERROR_CODES = {
    "fetch_error": "ERR001",
    "parse_error": "ERR002",
    "invalid_input": "ERR003",
    "server_error": "ERR004",
    "gcp_error": "ERR005"
}

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

# Initialize GCP Clients Globally with Thread-Safe Initialization
class GCPClients:
    gmail_service: Optional[Any] = None
    gcs_client: Optional[storage.Client] = None
    vision_client: Optional[vision.ImageAnnotatorClient] = None
    document_ai_client: Optional[documentai.DocumentProcessorServiceClient] = None
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def initialize_clients(cls, creds: Credentials):
        async with cls._lock:
            if cls.gmail_service is None:
                try:
                    cls.gmail_service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
                    logger.info("Gmail service initialized.", extra={"service": "gmail_service"})
                except Exception:
                    logger.error(
                        "Failed to initialize Gmail service.",
                        exc_info=True,
                        extra={"service": "gmail_service", "error": "Initialization failed."}
                    )
                    raise GCPError("Failed to initialize Gmail service.")

            if cls.gcs_client is None:
                try:
                    cls.gcs_client = storage.Client()
                    logger.info("GCS client initialized.", extra={"service": "storage.Client"})
                except Exception:
                    logger.error(
                        "Failed to initialize GCS client.",
                        exc_info=True,
                        extra={"service": "storage.Client", "error": "Initialization failed."}
                    )
                    raise GCPError("Failed to initialize GCS client.")

            if cls.vision_client is None:
                try:
                    cls.vision_client = vision.ImageAnnotatorClient()
                    logger.info("Vision client initialized.", extra={"service": "vision.ImageAnnotatorClient"})
                except Exception:
                    logger.error(
                        "Failed to initialize Vision client.",
                        exc_info=True,
                        extra={"service": "vision.ImageAnnotatorClient", "error": "Initialization failed."}
                    )
                    raise GCPError("Failed to initialize Vision client.")

            if cls.document_ai_client is None:
                try:
                    cls.document_ai_client = documentai.DocumentProcessorServiceClient()
                    logger.info("Document AI client initialized.", extra={"service": "DocumentProcessorServiceClient"})
                except Exception:
                    logger.error(
                        "Failed to initialize Document AI client.",
                        exc_info=True,
                        extra={"service": "DocumentProcessorServiceClient", "error": "Initialization failed."}
                    )
                    raise GCPError("Failed to initialize Document AI client.")

# Resilient API Call using built-in retries
async def resilient_api_call(func, *args, **kwargs):
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except HttpError as e:
        logger.error(
            "HTTP error during API call.",
            exc_info=True,
            extra={"error_code": e.resp.status}
        )
        raise GCPError(f"HTTP error: {e}")
    except Exception:
        logger.error(
            "Unexpected error during API call.",
            exc_info=True,
            extra={"error": "An unexpected error occurred."}
        )
        raise GCPError("Unexpected error during API call.")

# Gmail Service Initialization
async def get_gmail_service(creds: Credentials) -> Any:
    await GCPClients.initialize_clients(creds)
    if GCPClients.gmail_service is None:
        logger.error("Gmail service failed to initialize.", extra={"service": "gmail_service"})
        raise GCPError("Gmail service failed to initialize.")
    return GCPClients.gmail_service

# GCS Client Initialization
async def get_gcs_client() -> storage.Client:
    if GCPClients.gcs_client is None:
        logger.error("GCS client not initialized.", extra={"service": "storage.Client"})
        raise GCPError("GCS client not initialized.")
    return GCPClients.gcs_client

# Vision Client Initialization
async def get_vision_client() -> vision.ImageAnnotatorClient:
    if GCPClients.vision_client is None:
        logger.error("Vision client not initialized.", extra={"service": "vision.ImageAnnotatorClient"})
        raise GCPError("Vision client not initialized.")
    return GCPClients.vision_client

# Document AI Client Initialization
async def get_document_ai_client() -> documentai.DocumentProcessorServiceClient:
    if GCPClients.document_ai_client is None:
        logger.error("Document AI client not initialized.", extra={"service": "DocumentProcessorServiceClient"})
        raise GCPError("Document AI client not initialized.")
    return GCPClients.document_ai_client

# Decode Attachment
async def decode_attachment(service, message_id, attachment_id) -> bytes:
    try:
        attachment = await resilient_api_call(
            service.users().messages().attachments().get,
            userId='me',
            messageId=message_id,
            id=attachment_id
        )
        return base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
    except Exception:
        logger.error(
            f"{ERROR_CODES['fetch_error']}: Failed to decode attachment.",
            exc_info=True,
            extra={"message_id": message_id, "attachment_id": attachment_id, "error": "Decoding failed."}
        )
        raise FetchError("Failed to decode attachment.")

# Upload to GCS
async def upload_to_gcs(file_name: str, file_content: bytes) -> str:
    try:
        client = await get_gcs_client()
        bucket = client.bucket(settings.EMAIL_ATTACHMENT_BUCKET)
        blob = bucket.blob(file_name)
        await asyncio.to_thread(blob.upload_from_string, file_content)
        gcs_uri = f"gs://{settings.EMAIL_ATTACHMENT_BUCKET}/{file_name}"
        logger.info("Uploaded file to GCS.", extra={"gcs_uri": gcs_uri, "file_name": file_name})
        return gcs_uri
    except Exception:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Failed to upload file to GCS.",
            exc_info=True,
            extra={"file_name": file_name, "error": "Upload failed."}
        )
        raise GCPError(f"Failed to upload {file_name} to GCS.")

# MIME Type Validation
SUPPORTED_MIME_TYPES = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/png'
]

def validate_mime_type(mime_type: str):
    if mime_type not in SUPPORTED_MIME_TYPES and not mime_type.startswith('image/'):
        logger.warning("Unsupported MIME type detected.", extra={"mime_type": mime_type})
        raise GCPError(f"Unsupported MIME type: {mime_type}")

# Process PDF with Document AI
async def process_pdf(gcs_uri: str, mime_type: str) -> str:
    validate_mime_type(mime_type)
    try:
        client = await get_document_ai_client()
        name = f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.GCP_LOCATION}/processors/{settings.DOCUMENT_AI_PROCESSOR_ID}"
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
    except Exception:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Document AI processing failed for {gcs_uri}.",
            exc_info=True,
            extra={"gcs_uri": gcs_uri, "error": "Processing failed."}
        )
        raise GCPError(f"Document AI processing failed for {gcs_uri}.")

# Process Image with Vision AI
async def process_image(gcs_uri: str) -> str:
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
    except Exception:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Vision AI processing failed for {gcs_uri}.",
            exc_info=True,
            extra={"gcs_uri": gcs_uri, "error": "Processing failed."}
        )
        raise GCPError(f"Vision AI processing failed for {gcs_uri}.")

# Download from GCS
async def download_from_gcs(gcs_uri: str) -> bytes:
    try:
        client = await get_gcs_client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        blob = client.bucket(bucket_name).blob(blob_name)
        data = await asyncio.to_thread(blob.download_as_bytes)
        logger.info("Downloaded data from GCS.", extra={"gcs_uri": gcs_uri})
        return data
    except Exception:
        logger.error(
            f"Failed to download from GCS {gcs_uri}.",
            exc_info=True,
            extra={"gcs_uri": gcs_uri, "error": "Download failed."}
        )
        raise GCPError(f"Failed to download from GCS {gcs_uri}.")

# Process Attachment
async def process_attachment(attachment: Attachment) -> Optional[Dict[str, Any]]:
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
            logger.info(
                "Skipping unsupported attachment type.",
                extra={"mime_type": mime_type, "filename": filename}
            )
            return None
        return {"name": filename, "extracted_text": extracted_text}
    except Exception:
        logger.error(
            f"{ERROR_CODES['gcp_error']}: Failed to process attachment {filename}.",
            exc_info=True,
            extra={"filename": filename, "error": "Processing failed."}
        )
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
            'metadata': metadata,
            'body': body,
            'attachments': attachments,
        }
    except Exception:
        logger.error(
            "Error extracting message details.",
            exc_info=True,
            extra={"message_id": message.get('id'), "error": "Extraction failed."}
        )
        raise ParseError("Error extracting message details.")

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
        logger.info(
            "Combined thread messages.",
            extra={"thread_id": thread_id, "message_count": len(messages)}
        )
        return {
            "thread_id": thread_id,
            "subject": subject,
            "body": body,
            "attachments": deduped_attachments,
            "messages": messages,
        }
    except Exception:
        logger.error(
            "Error combining thread messages.",
            exc_info=True,
            extra={"thread_id": thread_id, "error": "Combining messages failed."}
        )
        raise ServerError("Failed to combine thread messages.")

# Fetch Thread
async def fetch_thread(thread_id: str, creds: Credentials) -> Dict[str, Any]:
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
    except Exception:
        logger.error(
            "Error fetching thread.",
            exc_info=True,
            extra={"thread_id": thread_id, "error": "Fetching thread failed."}
        )
        raise ServerError(f"Error fetching thread {thread_id}.")

# Fetch Email without Caching
async def fetch_email(email_id: str, creds: Credentials) -> Dict[str, Any]:
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
        return combined_thread
    except Exception:
        logger.error(
            "Failed to fetch email.",
            exc_info=True,
            extra={"email_id": email_id, "error": "Fetching email failed."}
        )
        raise FetchError(f"Failed to fetch email {email_id}.")

# Fetch Labels
async def fetch_labels(creds: Credentials) -> List[str]:
    service = await get_gmail_service(creds)
    try:
        response = await resilient_api_call(service.users().labels().list, userId='me')
        labels = [label['name'] for label in response.get('labels', [])]
        logger.info("Fetched labels.", extra={"label_count": len(labels)})
        return labels
    except Exception:
        logger.error(
            "Failed to fetch labels.",
            exc_info=True,
            extra={"error": "Fetching labels failed."}
        )
        raise FetchError("Failed to fetch labels.")

# Process Email
async def process_email(email_id: str, creds: Credentials) -> Dict[str, Any]:
    email = await fetch_email(email_id, creds)
    attachments = email.get('attachments', [])
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)

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
async def fetch_email_batch(email_ids: List[str], creds: Credentials) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
    results: List[Dict[str, Any]] = []

    async def handle_email(eid: str):
        async with semaphore:
            try:
                email_content = await process_email(eid, creds)
                results.append(email_content)
            except Exception:
                logger.error(
                    "Error processing email.",
                    exc_info=True,
                    extra={"email_id": eid, "error": "Processing failed."}
                )
                results.append({"error": f"Error processing email {eid}."})

    await asyncio.gather(*[handle_email(eid) for eid in email_ids])
    return results

# List Emails by Label
async def list_emails_by_label(label_name: str, creds: Credentials) -> List[str]:
    """
    Fetches email IDs that have the specified label.
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
    except Exception:
        logger.error(
            f"Failed to list emails by label '{label_name}'.",
            exc_info=True,
            extra={"label_name": label_name, "error": "Listing emails by label failed."}
        )
        raise FetchError(f"Failed to list emails by label '{label_name}'.")

# Process Emails by Label
async def process_emails_by_label(label_name: str, creds: Credentials) -> List[Dict[str, Any]]:
    """
    Fetches and processes emails that have the specified label.
    """
    try:
        email_ids = await list_emails_by_label(label_name, creds)
        if not email_ids:
            logger.info("No emails found for the specified label.", extra={"label_name": label_name})
            return []
        processed_emails = await fetch_email_batch(email_ids, creds)
        return processed_emails
    except Exception:
        logger.error(
            f"Failed to process emails by label '{label_name}'.",
            exc_info=True,
            extra={"label_name": label_name, "error": "Processing emails by label failed."}
        )
        raise FetchError(f"Failed to process emails by label '{label_name}'.")

# Insert Raw Parsed Output into BigQuery
async def insert_raw_parsed_output(
    email_id: str,
    raw_output: str,
    bigquery_client: Any,
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
    except Exception:
        logger.error(
            f"Failed to insert raw parsed output for email ID {email_id}.",
            exc_info=True,
            extra={"email_id": email_id, "error": "BigQuery insertion failed."}
        )
        raise ServerError(f"Failed to insert parsed data for email ID {email_id}.")


