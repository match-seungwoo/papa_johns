from __future__ import annotations


class AdapterError(Exception):
    """Base exception for all adapter errors."""


class AdapterConfigError(AdapterError):
    """Raised for missing or invalid configuration (e.g., empty API key)."""


class AdapterRequestError(AdapterError):
    """Raised when request parameters are invalid."""


class AdapterAPIError(AdapterError):
    """Raised when the vendor API returns an error response."""


class AdapterTimeoutError(AdapterError):
    """Raised when a vendor API call times out."""


class AdapterResponseError(AdapterError):
    """Raised when the vendor response is malformed or missing expected content."""
