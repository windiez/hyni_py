from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, Any
import json
import time
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue

# Type aliases
Headers = Dict[str, str]
JsonData = Dict[str, Any]
ProgressCallback = Optional[Callable[[], bool]]
StreamCallback = Callable[[str], None]
CompletionCallback = Optional[Callable[['HttpResponse'], None]]


@dataclass
class HttpResponse:
    """HTTP response data class"""
    success: bool
    status_code: int
    body: str
    headers: Dict[str, str]
    error_message: str = ""


class HttpMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class HttpClient(ABC):
    """Abstract base class for HTTP clients"""

    def __init__(self):
        self._headers: Headers = {}
        self._timeout: float = 30.0  # seconds
        self._max_retries: int = 3
        self._retry_delay: float = 1.0  # seconds

    def set_headers(self, headers: Headers) -> None:
        """Set default headers for all requests"""
        self._headers = headers.copy()

    def set_timeout(self, timeout_seconds: float) -> None:
        """Set request timeout in seconds"""
        self._timeout = timeout_seconds

    def set_max_retries(self, max_retries: int) -> None:
        """Set maximum number of retries for failed requests"""
        self._max_retries = max_retries

    @abstractmethod
    def post(self, url: str, data: JsonData,
             cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a POST request"""
        pass

    @abstractmethod
    def post_stream(self, url: str, data: JsonData,
                    on_chunk: StreamCallback,
                    on_complete: CompletionCallback = None,
                    cancel_check: ProgressCallback = None) -> None:
        """Send a POST request with streaming response"""
        pass

    @abstractmethod
    def get(self, url: str, params: Optional[Dict[str, str]] = None,
            cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a GET request"""
        pass


class RequestsHttpClient(HttpClient):
    """HTTP client implementation using the requests library"""

    def __init__(self):
        super().__init__()
        try:
            import requests
            self._requests = requests
            self._session = requests.Session()
            # Disable SSL verification to support enterprise environments with
            # corporate proxies, SSL inspection appliances, or self-signed CA chains.
            # The proxy layer is responsible for certificate validation in these setups.
            self._session.verify = False
        except ImportError:
            raise ImportError("requests library not installed. Install with: pip install requests")

    def __del__(self):
        if hasattr(self, '_session'):
            self._session.close()

    def post(self, url: str, data: JsonData,
             cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a POST request with retry logic"""
        headers = self._headers.copy()
        headers['Content-Type'] = 'application/json'

        last_error = None
        for attempt in range(self._max_retries):
            if cancel_check and cancel_check():
                return HttpResponse(
                    success=False,
                    status_code=0,
                    body="",
                    headers={},
                    error_message="Request cancelled"
                )

            try:
                response = self._session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self._timeout
                )

                return HttpResponse(
                    success=response.ok,
                    status_code=response.status_code,
                    body=response.text,
                    headers=dict(response.headers),
                    error_message="" if response.ok else f"HTTP {response.status_code}"
                )

            except self._requests.exceptions.Timeout:
                last_error = "Request timeout"
            except self._requests.exceptions.ConnectionError:
                last_error = "Connection error"
            except Exception as e:
                last_error = str(e)

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (attempt + 1))

        return HttpResponse(
            success=False,
            status_code=0,
            body="",
            headers={},
            error_message=last_error or "Unknown error"
        )

    def post_stream(self, url: str, data: JsonData,
                    on_chunk: StreamCallback,
                    on_complete: CompletionCallback = None,
                    cancel_check: ProgressCallback = None) -> None:
        """Send a POST request with streaming response"""
        headers = self._headers.copy()
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'text/event-stream'

        try:
            response = self._session.post(
                url,
                json=data,
                headers=headers,
                timeout=self._timeout,
                stream=True
            )

            if not response.ok:
                if on_complete:
                    on_complete(HttpResponse(
                        success=False,
                        status_code=response.status_code,
                        body=response.text,
                        headers=dict(response.headers),
                        error_message=f"HTTP {response.status_code}"
                    ))
                return

            # Process streaming response
            for line in response.iter_lines():
                if cancel_check and cancel_check():
                    break

                if line:
                    # Decode bytes to string
                    line_str = line.decode('utf-8')
                    on_chunk(line_str)

            if on_complete:
                on_complete(HttpResponse(
                    success=True,
                    status_code=response.status_code,
                    body="",  # Body already processed in chunks
                    headers=dict(response.headers)
                ))

        except Exception as e:
            if on_complete:
                on_complete(HttpResponse(
                    success=False,
                    status_code=0,
                    body="",
                    headers={},
                    error_message=str(e)
                ))

    def get(self, url: str, params: Optional[Dict[str, str]] = None,
            cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a GET request"""
        try:
            response = self._session.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self._timeout
            )

            return HttpResponse(
                success=response.ok,
                status_code=response.status_code,
                body=response.text,
                headers=dict(response.headers)
            )
        except Exception as e:
            return HttpResponse(
                success=False,
                status_code=0,
                body="",
                headers={},
                error_message=str(e)
            )


class HttpxHttpClient(HttpClient):
    """HTTP client implementation using the httpx library (async-capable)"""

    def __init__(self):
        super().__init__()
        try:
            import httpx
            self._httpx = httpx
            # Disable SSL verification consistent with RequestsHttpClient behaviour
            # for environments using corporate proxies or custom certificate chains.
            self._client = httpx.Client(verify=False)
        except ImportError:
            raise ImportError("httpx library not installed. Install with: pip install httpx")

    def __del__(self):
        if hasattr(self, '_client'):
            self._client.close()

    def post(self, url: str, data: JsonData,
             cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a POST request"""
        headers = self._headers.copy()
        headers['Content-Type'] = 'application/json'

        last_error = None
        for attempt in range(self._max_retries):
            if cancel_check and cancel_check():
                return HttpResponse(
                    success=False,
                    status_code=0,
                    body="",
                    headers={},
                    error_message="Request cancelled"
                )

            try:
                response = self._client.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self._timeout
                )

                return HttpResponse(
                    success=response.is_success,
                    status_code=response.status_code,
                    body=response.text,
                    headers=dict(response.headers),
                    error_message="" if response.is_success else f"HTTP {response.status_code}"
                )

            except self._httpx.TimeoutException:
                last_error = "Request timeout"
            except self._httpx.ConnectError:
                last_error = "Connection error"
            except Exception as e:
                last_error = str(e)

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (attempt + 1))

        return HttpResponse(
            success=False,
            status_code=0,
            body="",
            headers={},
            error_message=last_error or "Unknown error"
        )

    def post_stream(self, url: str, data: JsonData,
                    on_chunk: StreamCallback,
                    on_complete: CompletionCallback = None,
                    cancel_check: ProgressCallback = None) -> None:
        """Send a POST request with streaming response"""
        headers = self._headers.copy()
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'text/event-stream'

        try:
            with self._client.stream('POST', url, json=data, headers=headers) as response:
                if not response.is_success:
                    if on_complete:
                        on_complete(HttpResponse(
                            success=False,
                            status_code=response.status_code,
                            body=response.text,
                            headers=dict(response.headers),
                            error_message=f"HTTP {response.status_code}"
                        ))
                    return

                for line in response.iter_lines():
                    if cancel_check and cancel_check():
                        break

                    if line:
                        on_chunk(line)

                if on_complete:
                    on_complete(HttpResponse(
                        success=True,
                        status_code=response.status_code,
                        body="",
                        headers=dict(response.headers)
                    ))

        except Exception as e:
            if on_complete:
                on_complete(HttpResponse(
                    success=False,
                    status_code=0,
                    body="",
                    headers={},
                    error_message=str(e)
                ))

    def get(self, url: str, params: Optional[Dict[str, str]] = None,
            cancel_check: ProgressCallback = None) -> HttpResponse:
        """Send a GET request"""
        try:
            response = self._client.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self._timeout
            )

            return HttpResponse(
                success=response.is_success,
                status_code=response.status_code,
                body=response.text,
                headers=dict(response.headers)
            )
        except Exception as e:
            return HttpResponse(
                success=False,
                status_code=0,
                body="",
                headers={},
                error_message=str(e)
            )
