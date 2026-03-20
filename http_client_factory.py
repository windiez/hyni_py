from typing import Optional, Dict, Callable, Any, Union
from abc import ABC, abstractmethod

# Import from local module with relative import
from http_client import HttpClient, RequestsHttpClient, HttpxHttpClient, HttpResponse, StreamCallback


class HttpClientType:
    """Available HTTP client implementations"""
    REQUESTS = "requests"
    HTTPX = "httpx"
    AUTO = "auto"


class HttpClientFactory:
    """Factory for creating HTTP client instances"""

    @staticmethod
    def create_http_client(context: 'GeneralContext',
                          client_type: str | None = HttpClientType.AUTO) -> HttpClient:
        """
        Create an HTTP client instance

        Args:
            context: The general context (for future configuration needs)
            client_type: Type of client to create (auto, requests, httpx)

        Returns:
            An HTTP client instance

        Raises:
            ValueError: If specified client type is not available
        """
        if client_type == HttpClientType.AUTO:
            # Try requests first (most common), then httpx
            try:
                return RequestsHttpClient()
            except ImportError:
                try:
                    return HttpxHttpClient()
                except ImportError:
                    raise ValueError(
                        "No HTTP client library available. "
                        "Install either 'requests' or 'httpx': "
                        "pip install requests"
                    )

        elif client_type == HttpClientType.REQUESTS:
            try:
                return RequestsHttpClient()
            except ImportError:
                raise ValueError("requests library not installed. Install with: pip install requests")

        elif client_type == HttpClientType.HTTPX:
            try:
                return HttpxHttpClient()
            except ImportError:
                raise ValueError("httpx library not installed. Install with: pip install httpx")

        else:
            raise ValueError(f"Unknown HTTP client type: {client_type}")
