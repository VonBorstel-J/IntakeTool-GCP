#exceptions.py

class GmailUtilsException(Exception):
    """Base exception class for gmail_utils module."""
    pass

class FetchError(GmailUtilsException):
    """Exception raised for errors in fetching data."""
    pass

class ParseError(GmailUtilsException):
    """Exception raised for errors in parsing data."""
    pass

class InvalidInputError(GmailUtilsException):
    """Exception raised for invalid input."""
    pass

class ServerError(GmailUtilsException):
    """Exception raised for server errors."""
    pass

class GCPError(GmailUtilsException):
    """Exception raised for GCP-related errors."""
    pass

class GeminiAPIError(Exception):
    """Exception raised for Gemini API related errors."""
    pass

class PromptError(Exception):
    """Exception raised for prompt generation errors."""
    pass
