import asyncio
import json
from typing import Optional, Callable, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from dataclasses import dataclass
from enum import Enum

# Import HttpResponse from http_client
from http_client import HttpResponse

# Type aliases for callbacks
ProgressCallback = Optional[Callable[[], bool]]
StreamCallback = Callable[[str], None]
CompletionCallback = Optional[Callable[['HttpResponse'], None]]


class ChatApiError(RuntimeError):
    """Base exception for chat API errors"""
    pass


class StreamingNotSupportedError(ChatApiError):
    """Raised when streaming is requested but not supported by provider"""
    def __init__(self):
        super().__init__("Streaming is not supported by this provider")


class NoUserMessageError(ChatApiError):
    """Raised when no user message is found in context"""
    def __init__(self):
        super().__init__("No user message found in context")


class FailedApiResponse(ChatApiError):
    """Raised when API response parsing fails"""
    def __init__(self, message: str):
        super().__init__(f"Failed to parse API response: {message}")


class ChatApi:
    """
    Main class for interacting with LLM chat APIs
    """

    def __init__(self, context: 'GeneralContext'):
        """
        Constructs a chat API with the given context

        Args:
            context: The general context to use for API interactions
        """
        self._context = context
        self._http_client = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ensure_http_client()

    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def send_message(self, message: Optional[str] = None,
                     cancel_check: ProgressCallback = None) -> str:
        """
        Sends a message and waits for a response

        Args:
            message: The message to send. If None, uses existing context messages
            cancel_check: Optional callback to check if operation should be cancelled

        Returns:
            The response text

        Raises:
            NoUserMessageError: If no user message exists when message is None
            FailedApiResponse: If the request fails or response cannot be parsed
        """
        self._ensure_http_client()

        if message is not None:
            # Clear previous messages and add new one
            self._context.clear_user_messages()
            self._context.add_user_message(message)
        else:
            # Validate we have at least one user message
            has_user_message = any(
                msg.get("role") == "user"
                for msg in self._context.get_messages()
            )
            if not has_user_message:
                raise NoUserMessageError()

        request = self._context.build_request()
        self._http_client.set_headers(self._context.get_headers())
        response = self._http_client.post(
            self._context.get_endpoint(),
            request,
            cancel_check
        )

        if not response.success:
            # Include provider context and a key suffix to help identify which
            # credential failed in deployments that rotate or pool multiple API keys.
            # Only the last 8 characters are included — insufficient to reconstruct
            # the full key but enough to match against a known key inventory.
            key_diag = ""
            if self._context._api_key:
                key_diag = f", key=...{self._context._api_key[-8:]}"
            raise FailedApiResponse(f"{response.error_message} [provider={self._context.get_provider_name()}{key_diag}]")

        try:
            json_response = json.loads(response.body)
            return self._context.extract_text_response(json_response)
        except Exception as e:
            raise FailedApiResponse(str(e))

    def send_message_stream(self,
                           message: Optional[str] = None,
                           on_chunk: StreamCallback = None,
                           on_complete: CompletionCallback = None,
                           cancel_check: ProgressCallback = None) -> None:
        """
        Sends a message and streams the response

        Args:
            message: The message to send. If None, uses existing context messages
            on_chunk: Callback function to handle each chunk of the response
            on_complete: Optional callback function to handle completion
            cancel_check: Optional callback to check if operation should be cancelled

        Raises:
            StreamingNotSupportedError: If streaming is not supported
            NoUserMessageError: If no user message exists when message is None
        """
        self._ensure_http_client()

        if not self._context.supports_streaming():
            raise StreamingNotSupportedError()

        if message is not None:
            self._context.clear_user_messages()
            self._context.add_user_message(message)
        else:
            # Validate we have at least one user message
            has_user_message = any(
                msg.get("role") == "user"
                for msg in self._context.get_messages()
            )
            if not has_user_message:
                raise NoUserMessageError()

        # Build request with streaming enabled
        request = self._context.build_request(streaming=True)
        self._http_client.set_headers(self._context.get_headers())

        # Wrap the chunk handler to parse SSE format
        def chunk_wrapper(chunk: str):
            self._parse_stream_chunk(chunk, on_chunk)

        self._http_client.post_stream(
            self._context.get_endpoint(),
            request,
            chunk_wrapper,
            on_complete,
            cancel_check
        )

    def send_message_async(self, message: Optional[str] = None) -> Future[str]:
        """
        Sends a message asynchronously

        Args:
            message: The message to send. If None, uses existing context messages

        Returns:
            A Future containing the response text
        """
        # Use thread pool executor for async operations
        return self._executor.submit(self.send_message, message)

    async def send_message_async_await(self, message: Optional[str] = None) -> str:
        """
        Sends a message asynchronously (async/await style)

        Args:
            message: The message to send. If None, uses existing context messages

        Returns:
            The response text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.send_message,
            message
        )

    @property
    def context(self) -> 'GeneralContext':
        """Gets the underlying context for advanced usage"""
        return self._context

    def _parse_stream_chunk(self, chunk: str, on_chunk: StreamCallback) -> None:
        """
        Parses Server-Sent Events (SSE) formatted streaming chunks

        This method processes streaming responses that follow the SSE format:
        - Lines starting with "data: " contain the actual JSON payload
        - Empty lines are separators between events
        - The special "[DONE]" message indicates end of stream

        Args:
            chunk: Raw chunk received from the HTTP stream
            on_chunk: Callback to invoke with extracted content
        """
        try:
            lines = chunk.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # SSE format: lines starting with "data: " contain the payload
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix

                    # Special marker indicating end of stream
                    if json_str == "[DONE]":
                        return

                    try:
                        json_chunk = json.loads(json_str)
                        content = self._context.extract_text_response(json_chunk)
                        if content:
                            on_chunk(content)
                    except json.JSONDecodeError:
                        # Silently skip malformed JSON chunks
                        pass
        except Exception:
            # Catch any unexpected errors to prevent callback interruption
            pass

    def _ensure_http_client(self) -> None:
        """
        Ensures that the HTTP client is initialized
        """
        if self._http_client is None:
            # Fix import path
            from http_client_factory import HttpClientFactory
            self._http_client = HttpClientFactory.create_http_client(self._context)


class SchemaState(Enum):
    """Builder state for type safety"""
    NEEDS_SCHEMA = "needs_schema"
    HAS_SCHEMA = "has_schema"


class ChatApiBuilder:
    """
    Builder for creating ChatApi instances with fluent interface

    Example:
        api = (ChatApiBuilder()
               .schema("schemas/claude.json")
               .api_key(key)
               .config(config)
               .build())
    """

    def __init__(self, state: SchemaState = SchemaState.NEEDS_SCHEMA):
        self._state = state
        self._schema_path = None
        self._config = None
        self._api_key = None
        self._timeout = 30000  # milliseconds
        self._max_retries = 3

    def schema(self, path: str) -> 'ChatApiBuilder':
        """Sets the schema path (required first)"""
        if self._state != SchemaState.NEEDS_SCHEMA:
            raise ValueError("Schema already set")

        new_builder = ChatApiBuilder(SchemaState.HAS_SCHEMA)
        new_builder._schema_path = path
        new_builder._config = self._config
        new_builder._api_key = self._api_key
        new_builder._timeout = self._timeout
        new_builder._max_retries = self._max_retries
        return new_builder

    def config(self, cfg: 'ContextConfig') -> 'ChatApiBuilder':
        """Sets the context configuration"""
        self._config = cfg
        return self

    def api_key(self, key: str) -> 'ChatApiBuilder':
        """Sets the API key"""
        self._api_key = key
        return self

    def timeout(self, timeout_ms: int) -> 'ChatApiBuilder':
        """Sets the timeout in milliseconds"""
        self._timeout = timeout_ms
        return self

    def max_retries(self, retries: int) -> 'ChatApiBuilder':
        """Sets the maximum number of retries"""
        self._max_retries = retries
        return self

    def build(self) -> ChatApi:
        """Builds the ChatApi instance"""
        if self._state != SchemaState.HAS_SCHEMA:
            raise ValueError("Schema must be set before building")

        from general_context import GeneralContext

        context = GeneralContext(self._schema_path, self._config)
        if self._api_key:
            context.set_api_key(self._api_key)

        # TODO: Pass timeout and max_retries to HTTP client
        return ChatApi(context)


# Usage example:
if __name__ == "__main__":
    # Example usage
    from general_context import ContextConfig

    config = ContextConfig(
        enable_validation=True,
        default_temperature=0.7,
        default_max_tokens=1000
    )

    # Create API instance using builder
    api = (ChatApiBuilder()
           .schema("schemas/openai.json")
           .config(config)
           .api_key("your-api-key")
           .timeout(60000)
           .max_retries(3)
           .build())

    # Synchronous usage
    response = api.send_message("Hello, how are you?")
    print(response)

    # Async usage
    future = api.send_message_async("Tell me a joke")
    response = future.result()
    print(response)

    # Streaming usage
    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)

    def on_complete(response: HttpResponse):
        print("\nStreaming complete!")

    api.send_message_stream(
        "Write a short story",
        on_chunk=on_chunk,
        on_complete=on_complete
    )
