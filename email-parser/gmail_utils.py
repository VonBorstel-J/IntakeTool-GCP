#gmail_utils.py
import base64, os, logging, asyncio
from datetime import datetime
from typing import List, Dict, Any, Callable
from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("gmail_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
ERROR_CODES = {"fetch_error": "ERR001", "parse_error": "ERR002", "invalid_input": "ERR003", "server_error": "ERR004"}
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service() -> Any:
    try:
        service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
        if service_account_path and os.path.exists(service_account_path):
            credentials = service_account.Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
            logger.info("Using service account credentials.")
        else:
            credentials, _ = default(scopes=SCOPES)
            logger.info("Using default credentials.")
        return build('gmail', 'v1', credentials=credentials)
    except Exception as e:
        logger.error(f"{ERROR_CODES['fetch_error']}: Failed to create Gmail service. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to initialize Gmail service."}})

def decode_body(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data).decode('utf-8')
    except Exception as e:
        logger.error(f"{ERROR_CODES['parse_error']}: Failed to decode email body. Cause: {str(e)}")
        return ""

@retry(retry=retry_if_exception_type(HttpError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def fetch_with_retry(service_method: Callable[..., Any], *args, **kwargs) -> Any:
    try:
        return service_method(*args, **kwargs).execute()
    except HttpError as e:
        if e.resp.status in [500, 502, 503, 504]:
            logger.warning(f"{ERROR_CODES['fetch_error']}: Transient error encountered: {e}. Retrying...")
            raise
        else:
            logger.error(f"{ERROR_CODES['fetch_error']}: Non-retriable error encountered: {e}")
            raise
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Unexpected error in fetch_with_retry. Cause: {str(e)}")
        raise

def decode_attachment(service: Any, message_id: str, attachment_id: str) -> Dict[str, Any]:
    try:
        attachment = fetch_with_retry(service.users().messages().attachments().get, userId='me', messageId=message_id, id=attachment_id)
        if 'data' not in attachment:
            logger.warning(f"{ERROR_CODES['parse_error']}: Missing 'data' in attachment {attachment_id} for message {message_id}.")
            return {'filename': 'UNKNOWN_FILENAME', 'data': None, 'error': 'Attachment decoding failed'}
        data = base64.urlsafe_b64decode(attachment['data'])
        return {'filename': 'UNKNOWN_FILENAME', 'data': data}
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Error decoding attachment {attachment_id} for message {message_id}. Cause: {str(e)}")
        return {'filename': 'UNKNOWN_FILENAME', 'data': None, 'error': str(e)}

def extract_message_details(service: Any, message: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = message.get('payload', {})
        if not payload:
            logger.warning(f"{ERROR_CODES['parse_error']}: Message {message.get('id')} has no payload.")
            return {}
        headers = payload.get('headers', []) or []
        subject = next((header['value'] for header in headers if header.get('name') == 'Subject' and header.get('value').strip()), '')
        body = ""
        if 'data' in payload.get('body', {}):
            body = decode_body(payload['body']['data'])
        else:
            parts = payload.get('parts', [])
            for part in parts or []:
                if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                    body = decode_body(part['body']['data'])
                    break
        attachments = []
        parts = payload.get('parts', [])
        for part in parts or []:
            filename = part.get('filename')
            attachment_id = part.get('body', {}).get('attachmentId')
            if filename and attachment_id:
                attachment = decode_attachment(service, message['id'], attachment_id)
                if attachment and attachment.get('data'):
                    attachment['filename'] = filename
                    attachments.append(attachment)
                else:
                    logger.warning(f"{ERROR_CODES['parse_error']}: Skipping attachment {filename} for message {message.get('id')} due to decoding failure.")
        from_header = next((header['value'] for header in headers if header.get('name') == 'From' and header.get('value').strip()), '')
        timestamp = message.get('internalDate')
        if timestamp:
            try:
                timestamp = datetime.utcfromtimestamp(int(timestamp)/1000).isoformat() + 'Z'
            except (ValueError, OSError) as e:
                logger.error(f"{ERROR_CODES['parse_error']}: Invalid timestamp format for message {message.get('id')}. Cause: {str(e)}")
                timestamp = ""
        return {'message_id': message.get('id'), 'from': from_header, 'timestamp': timestamp, 'subject': subject, 'body': body, 'attachments': attachments}
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Error extracting details from message {message.get('id')}. Cause: {str(e)}")
        return {}

def combine_thread_messages(thread_id: str, processed_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        body_parts = []
        combined_attachments = []
        subject = ""
        for msg in processed_messages:
            if not subject and msg.get('subject'):
                subject = msg['subject']
            body_parts.append(msg.get('body', ''))
            combined_attachments.extend(msg.get('attachments', []))
        combined_body = "\n\n".join(body_parts).strip()
        return {"thread_id": thread_id, "subject": subject.strip(), "body": combined_body, "attachments": combined_attachments, "messages": processed_messages}
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Error combining thread messages for thread {thread_id}. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Failed to combine thread messages."}})

async def fetch_email(email_id: str) -> Dict[str, Any]:
    service = get_gmail_service()
    try:
        logger.info(f"{ERROR_CODES['fetch_error']}: Fetching email with ID: {email_id}")
        message = await asyncio.to_thread(fetch_with_retry, service.users().messages().get, 'me', email_id, {'format': 'full'})
        thread_id = message.get('threadId')
        if not thread_id:
            error_msg = f"{ERROR_CODES['fetch_error']}: Thread ID not found for email ID: {email_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": error_msg}})
        logger.info(f"Email ID: {email_id} belongs to thread ID: {thread_id}")
    except HttpError as e:
        logger.error(f"{ERROR_CODES['fetch_error']}: HTTP error while fetching email {email_id}. Cause: {str(e)}")
        raise HTTPException(status_code=e.resp.status, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to fetch email due to an HTTP error."}})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Unexpected error while fetching email {email_id}. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error while fetching email."}})
    try:
        logger.info(f"{ERROR_CODES['fetch_error']}: Fetching thread with ID: {thread_id}")
        thread = await asyncio.to_thread(fetch_with_retry, service.users().threads().get, 'me', thread_id, {'format': 'full'})
        messages = thread.get('messages', []) or []
        if not messages:
            error_msg = f"{ERROR_CODES['fetch_error']}: No messages found in thread ID: {thread_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": error_msg}})
        sorted_messages = sorted(messages, key=lambda msg: int(msg.get('internalDate', '0')))
        logger.info(f"Fetched {len(sorted_messages)} messages in thread ID: {thread_id}")
    except HttpError as e:
        logger.error(f"{ERROR_CODES['fetch_error']}: HTTP error while fetching thread {thread_id}. Cause: {str(e)}")
        raise HTTPException(status_code=e.resp.status, detail={"error": {"code": ERROR_CODES['fetch_error'], "message": "Failed to fetch thread due to an HTTP error."}})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{ERROR_CODES['server_error']}: Unexpected error while fetching thread {thread_id}. Cause: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error while fetching thread."}})
    processed_messages = []
    for msg in sorted_messages:
        details = extract_message_details(service, msg)
        if details:
            processed_messages.append(details)
    if not processed_messages:
        error_msg = f"{ERROR_CODES['parse_error']}: All messages in thread ID: {thread_id} are malformed or empty."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail={"error": {"code": ERROR_CODES['parse_error'], "message": error_msg}})
    combined_result = combine_thread_messages(thread_id, processed_messages)
    return combined_result

async def fetch_email_batch(email_ids: List[str]) -> List[Dict[str, Any]]:
    tasks = [fetch_email(email_id) for email_id in email_ids]
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            email_data = await task
            results.append(email_data)
        except HTTPException as he:
            logger.error(f"{he.detail['error']['code']}: Error fetching email in batch. Cause: {he.detail['error']['message']}")
            results.append({"error": he.detail["error"]})
        except Exception as e:
            logger.error(f"{ERROR_CODES['server_error']}: Error fetching email in batch. Cause: {str(e)}")
            results.append({"error": {"code": ERROR_CODES['server_error'], "message": "Unexpected error during batch email fetching."}})
    return results
