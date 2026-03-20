"""
General Context Module for LLM API Interactions

This module provides a schema-based approach to handling context and interactions
with various Language Model APIs. It supports multiple providers through JSON
schema configuration files.

Note: This class is NOT thread-safe. In multi-threaded environments, each thread
should maintain its own instance. Consider using thread-local storage.
"""

import json
import base64
import os
from typing import Dict, List, Optional, Any, Union, TypeVar, Type, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

T = TypeVar('T')


class SchemaException(Exception):
    """Custom exception for schema-related errors."""

    def __init__(self, message: str) -> None:
        """
        Initialize a schema exception.

        Args:
            message: The error message describing the schema issue
        """
        super().__init__(message)
        self.message = message


class ValidationException(Exception):
    """Custom exception for validation-related errors."""

    def __init__(self, message: str) -> None:
        """
        Initialize a validation exception.

        Args:
            message: The error message describing the validation issue
        """
        super().__init__(message)
        self.message = message


@dataclass
class ContextConfig:
    """Configuration structure for additional context options."""

    enable_streaming_support: bool = False
    """Whether to enable streaming support"""

    enable_validation: bool = True
    """Whether to enable validation"""

    enable_caching: bool = True
    """Whether to enable caching"""

    default_max_tokens: Optional[int] = None
    """Default maximum tokens for responses"""

    default_temperature: Optional[float] = None
    """Default temperature for responses"""

    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    """Custom parameters"""


def remove_nulls_recursive(data: Any) -> Any:
    """
    Recursively remove null values from a dictionary or list.

    Args:
        data: The data structure to clean

    Returns:
        The cleaned data structure with null values removed
    """
    if isinstance(data, dict):
        return {k: remove_nulls_recursive(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_nulls_recursive(item) for item in data if item is not None]
    else:
        return data


class GeneralContext:
    """
    Main class for handling LLM context and API interactions.

    This class manages the context for interacting with language model APIs,
    including message handling, parameter configuration, and request/response processing.

    The class uses JSON schema files to define provider-specific configurations,
    making it adaptable to different API formats and requirements.

    Note: This class is NOT thread-safe. In multi-threaded environments, each thread
    should maintain its own instance.
    """

    def __init__(self, schema_input: Union[str, Dict[str, Any]],
                 config: Optional[ContextConfig] = None) -> None:
        """
        Initialize a general context with the given schema and configuration.

        Args:
            schema_input: Either a path to the schema file (str) or a pre-loaded schema (dict)
            config: Configuration options

        Raises:
            SchemaException: If the schema is invalid or cannot be loaded
        """
        self._config = config or ContextConfig()

        # Core data members
        self._schema: Dict[str, Any] = {}
        self._request_template: Dict[str, Any] = {}

        # Provider information
        self._provider_name: str = ""
        self._endpoint: str = ""
        self._headers: Dict[str, str] = {}
        self._model_name: str = ""
        self._system_message: Optional[str] = None
        self._messages: List[Dict[str, Any]] = []
        self._parameters: Dict[str, Any] = {}
        self._api_key: str = ""
        self._valid_roles: Set[str] = set()

        # Cached paths and formats
        self._text_path: List[str] = []
        self._error_path: List[str] = []
        self._message_structure: Dict[str, Any] = {}
        self._text_content_format: Dict[str, Any] = {}
        self._image_content_format: Dict[str, Any] = {}

        # Initialize from schema
        if isinstance(schema_input, str):
            self._load_schema(schema_input)
        else:
            self._schema = schema_input.copy()

        self._validate_schema()
        self._cache_schema_elements()
        self._apply_defaults()
        self._build_headers()

    def _load_schema(self, schema_path: str) -> None:
        """
        Load schema from file.

        Args:
            schema_path: Path to the schema file

        Raises:
            SchemaException: If the schema file cannot be loaded or parsed
        """
        try:
            with open(schema_path, 'r', encoding='utf-8') as file:
                self._schema = json.load(file)
        except FileNotFoundError:
            raise SchemaException(f"Failed to open schema file: {schema_path}")
        except json.JSONDecodeError as e:
            raise SchemaException(f"Failed to parse schema JSON: {str(e)}")

    def _validate_schema(self) -> None:
        """
        Validate the loaded schema structure.

        Raises:
            SchemaException: If required schema fields are missing
        """
        required_fields = ["provider", "api", "request_template", "message_format", "response_format"]

        for field in required_fields:
            if field not in self._schema:
                raise SchemaException(f"Missing required schema field: {field}")

        # Validate API configuration
        if "endpoint" not in self._schema["api"]:
            raise SchemaException("Missing API endpoint in schema")

        # Validate message format
        message_format = self._schema["message_format"]
        if "structure" not in message_format or "content_types" not in message_format:
            raise SchemaException("Invalid message format in schema")

        # Validate response format
        response_format = self._schema["response_format"]
        if ("success" not in response_format or
            "text_path" not in response_format["success"]):
            raise SchemaException("Invalid response format in schema")

    def _cache_schema_elements(self) -> None:
        """Cache frequently accessed schema elements for performance."""
        # Cache provider info
        self._provider_name = self._schema["provider"]["name"]
        self._endpoint = self._schema["api"]["endpoint"]

        if "message_roles" in self._schema:
            self._valid_roles = set(self._schema["message_roles"])

        # Cache request template
        self._request_template = self._schema["request_template"].copy()

        # Cache response paths
        self._text_path = self._parse_json_path(self._schema["response_format"]["success"]["text_path"])
        if ("error" in self._schema["response_format"] and
            "error_path" in self._schema["response_format"]["error"]):
            self._error_path = self._parse_json_path(self._schema["response_format"]["error"]["error_path"])

        # Cache message formats
        self._message_structure = self._schema["message_format"]["structure"].copy()
        content_types = self._schema["message_format"]["content_types"]
        if "text" in content_types:
            self._text_content_format = content_types["text"].copy()
        if "image" in content_types:
            self._image_content_format = content_types["image"].copy()

    def _build_headers(self) -> None:
        """Build HTTP headers for API requests."""
        self._headers.clear()

        # Process required headers
        if "headers" in self._schema and "required" in self._schema["headers"]:
            for key, value in self._schema["headers"]["required"].items():
                header_value = str(value)

                if ("authentication" in self._schema and
                    "key_placeholder" in self._schema["authentication"]):

                    placeholder = self._schema["authentication"]["key_placeholder"]

                    # Check if the header value already contains the prefix
                    if "key_prefix" in self._schema["authentication"]:
                        prefix = self._schema["authentication"]["key_prefix"]

                        # If the placeholder in the header already has the prefix, don't add it again
                        if prefix in header_value and header_value.find(prefix) < header_value.find(placeholder):
                            # The prefix is already in the header template, just replace placeholder with key
                            replacement = self._api_key
                        else:
                            # The prefix is not in the header template, add it
                            replacement = prefix + self._api_key
                    else:
                        replacement = self._api_key

                    # Replace all occurrences of placeholder
                    header_value = header_value.replace(placeholder, replacement)

                self._headers[key] = header_value

        # Process optional headers (only if values are provided)
        if "headers" in self._schema and "optional" in self._schema["headers"]:
            for key, value in self._schema["headers"]["optional"].items():
                if value is not None and isinstance(value, str) and value.strip():
                    self._headers[key] = str(value)

    def _apply_defaults(self) -> None:
        """Apply default values from schema."""
        if "models" in self._schema and "default" in self._schema["models"]:
            self._model_name = self._schema["models"]["default"]

    def set_model(self, model: str) -> 'GeneralContext':
        """
        Set the model to use for requests.

        Args:
            model: The model name

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the model is not supported and validation is enabled
        """
        # Validate model if available models are specified
        if ("models" in self._schema and "available" in self._schema["models"]):
            available_models = self._schema["models"]["available"]
            if model not in available_models and self._config.enable_validation:
                raise ValidationException(f"Model '{model}' is not supported by this provider")

        self._model_name = model
        return self

    def set_system_message(self, system_text: str) -> 'GeneralContext':
        """
        Set the system message for the conversation.

        Args:
            system_text: The system message text

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If system messages are not supported and validation is enabled
        """
        if not self.supports_system_messages() and self._config.enable_validation:
            raise ValidationException(f"Provider '{self._provider_name}' does not support system messages")

        self._system_message = system_text
        return self

    def set_parameter(self, key: str, value: Any) -> 'GeneralContext':
        """
        Set a single parameter for the request.

        Args:
            key: The parameter key
            value: The parameter value

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the parameter is invalid and validation is enabled
        """
        if self._config.enable_validation:
            self._validate_parameter(key, value)

        self._parameters[key] = value
        return self

    def set_parameters(self, params: Dict[str, Any]) -> 'GeneralContext':
        """
        Set multiple parameters for the request.

        Args:
            params: Dictionary of parameter keys and values

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If any parameter is invalid and validation is enabled
        """
        for key, value in params.items():
            self.set_parameter(key, value)
        return self

    def set_api_key(self, api_key: str) -> 'GeneralContext':
        """
        Set the API key for authentication.

        Args:
            api_key: The API key

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the API key is empty
        """
        if not api_key:
            raise ValidationException("API key cannot be empty")

        self._api_key = api_key
        self._build_headers()  # Rebuild headers with new API key
        return self

    def add_user_message(self, content: str, media_type: Optional[str] = None,
                        media_data: Optional[str] = None) -> 'GeneralContext':
        """
        Add a user message to the conversation.

        Args:
            content: The message content
            media_type: Optional media type for multimodal content
            media_data: Optional media data for multimodal content

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the message is invalid and validation is enabled
        """
        return self.add_message("user", content, media_type, media_data)

    def add_assistant_message(self, content: str) -> 'GeneralContext':
        """
        Add an assistant message to the conversation.

        Args:
            content: The message content

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the message is invalid and validation is enabled
        """
        return self.add_message("assistant", content)

    def add_message(self, role: str, content: str, media_type: Optional[str] = None,
                   media_data: Optional[str] = None) -> 'GeneralContext':
        """
        Add a message with the specified role to the conversation.

        Args:
            role: The message role (e.g., "user", "assistant", "system")
            content: The message content
            media_type: Optional media type for multimodal content
            media_data: Optional media data for multimodal content

        Returns:
            Reference to this context for method chaining

        Raises:
            ValidationException: If the message is invalid and validation is enabled
        """
        message = self._create_message(role, content, media_type, media_data)
        if self._config.enable_validation:
            self._validate_message(message)

        self._messages.append(message)
        return self

    def _create_message(self, role: str, content: str, media_type: Optional[str] = None,
                       media_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a message object with the specified parameters.

        Args:
            role: The message role
            content: The message content
            media_type: Optional media type for multimodal content
            media_data: Optional media data for multimodal content

        Returns:
            Dictionary representing the message

        Raises:
            ValidationException: If multimodal content is not supported
        """
        message = self._message_structure.copy()
        message["role"] = role

        # Create content array
        content_array = [self._create_text_content(content)]

        # Add image if provided
        if media_type and media_data:
            if not self.supports_multimodal() and self._config.enable_validation:
                raise ValidationException(f"Provider '{self._provider_name}' does not support multimodal content")
            content_array.append(self._create_image_content(media_type, media_data))

        message["content"] = content_array
        return message

    def _create_text_content(self, text: str) -> Dict[str, Any]:
        """
        Create a text content object.

        Args:
            text: The text content

        Returns:
            Dictionary representing the text content
        """
        content = self._text_content_format.copy()
        content["text"] = text
        return content

    def _create_image_content(self, media_type: str, data: str) -> Dict[str, Any]:
        """
        Create an image content object.

        Args:
            media_type: The media type (e.g., "image/jpeg")
            data: The image data (base64 encoded or file path)

        Returns:
            Dictionary representing the image content
        """
        content = self._image_content_format.copy()
        content["source"]["media_type"] = media_type

        # Handle both base64 data and file paths
        if self._is_base64_encoded(data):
            content["source"]["data"] = data
        else:
            # Assume it's a file path and encode it
            content["source"]["data"] = self._encode_image_to_base64(data)

        return content

    def build_request(self, streaming: bool = False) -> Dict[str, Any]:
        """
        Build a request object based on the current context.

        Args:
            streaming: Whether to enable streaming for this request

        Returns:
            Dictionary representing the API request
        """
        # The request template is populated once at schema load time and is
        # read-only thereafter — copying it every call is unnecessary overhead.
        request = self._request_template
        messages = list(self._messages)  # Messages still need a copy for potential insertion

        # Set model
        if self._model_name:
            request["model"] = self._model_name

        # Set system message if supported
        if self._system_message and self.supports_system_messages():
            system_in_roles = "system" in self._valid_roles

            if system_in_roles:
                # OpenAI style - system role supported in messages
                system_msg = self._create_message("system", self._system_message)
                messages.insert(0, system_msg)
            else:
                # Claude style - use separate system field
                request["system"] = self._system_message

        # Set messages
        request["messages"] = messages

        # Apply custom parameters FIRST (so they take precedence)
        for key, value in self._parameters.items():
            request[key] = value

        # Apply config defaults only if not already set
        if self._config.default_max_tokens and "max_tokens" not in request:
            request["max_tokens"] = self._config.default_max_tokens
        if self._config.default_temperature and "temperature" not in request:
            request["temperature"] = self._config.default_temperature

        # Set streaming: user parameter takes precedence over function parameter
        if "stream" not in self._parameters:
            # User hasn't explicitly set stream parameter, use function parameter
            if streaming and self._schema.get("features", {}).get("streaming", False):
                request["stream"] = True
            else:
                request["stream"] = False

        return remove_nulls_recursive(request)

    def extract_text_response(self, response: Dict[str, Any]) -> str:
        """
        Extract the text response from a JSON response.

        Args:
            response: The JSON response from the API

        Returns:
            The extracted text

        Raises:
            RuntimeError: If the text cannot be extracted
        """
        try:
            text_node = self._resolve_path(response, self._text_path)
            return str(text_node)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text response: {str(e)}")

    def extract_full_response(self, response: Dict[str, Any]) -> Any:
        """
        Extract the full response content from a JSON response.

        Args:
            response: The JSON response from the API

        Returns:
            The extracted content

        Raises:
            RuntimeError: If the content cannot be extracted
        """
        try:
            content_path = self._parse_json_path(self._schema["response_format"]["success"]["content_path"])
            return self._resolve_path(response, content_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract full response: {str(e)}")

    def extract_error(self, response: Dict[str, Any]) -> str:
        """
        Extract an error message from a JSON response.

        Args:
            response: The JSON response from the API

        Returns:
            The extracted error message
        """
        if not self._error_path:
            return "Unknown error"

        try:
            error_node = self._resolve_path(response, self._error_path)
            return str(error_node)
        except Exception:
            return "Failed to parse error message"

    def reset(self) -> None:
        """
        Reset the context to its initial state.

        Raises:
            json.JSONDecodeError: If the initial content cannot be parsed
        """
        self.clear_user_messages()
        self.clear_system_message()
        self.clear_parameters()
        self._model_name = ""
        self._apply_defaults()

    def clear_user_messages(self) -> None:
        """Clear all user messages in the context."""
        self._messages.clear()

    def clear_system_message(self) -> None:
        """Clear system message in the context."""
        self._system_message = None

    def clear_parameters(self) -> None:
        """Clear all parameters in the context."""
        self._parameters.clear()

    def has_api_key(self) -> bool:
        """
        Check if an API key has been set.

        Returns:
            True if a non-empty API key has been configured, False otherwise.
        """
        return bool(self._api_key)

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema used by this context.

        Returns:
            The schema as a dictionary
        """
        return self._schema.copy()

    def get_provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return self._provider_name

    def get_endpoint(self) -> str:
        """
        Get the API endpoint.

        Returns:
            The API endpoint URL
        """
        return self._endpoint

    def get_headers(self) -> Dict[str, str]:
        """
        Get the HTTP headers for API requests.

        Returns:
            Dictionary of header names to values
        """
        return self._headers.copy()

    def get_supported_models(self) -> List[str]:
        """
        Get the list of models supported by the provider.

        Returns:
            List of supported model names
        """
        models = []
        if "models" in self._schema and "available" in self._schema["models"]:
            models = list(self._schema["models"]["available"])
        return models

    def supports_multimodal(self) -> bool:
        """
        Check if the provider supports multimodal content.

        Returns:
            True if multimodal content is supported, False otherwise
        """
        multimodal = self._schema.get("multimodal")
        if multimodal and isinstance(multimodal, dict):
            supported = multimodal.get("supported")
            if supported is not None and isinstance(supported, bool):
                return supported
        return False

    def supports_streaming(self) -> bool:
        """
        Check if the provider supports streaming.

        Returns:
            True if streaming is supported, False otherwise
        """
        features = self._schema.get("features")
        if features and isinstance(features, dict):
            streaming = features.get("streaming")
            if streaming is not None and isinstance(streaming, bool):
                return streaming
        return False

    def supports_system_messages(self) -> bool:
        """
        Check if the provider supports system messages.

        Returns:
            True if system messages are supported, False otherwise
        """
        system_message = self._schema.get("system_message")
        if system_message and isinstance(system_message, dict):
            supported = system_message.get("supported")
            if supported is not None and isinstance(supported, bool):
                return supported
        return False

    def is_valid_request(self) -> bool:
        """
        Check if the current context would produce a valid request.

        Returns:
            True if the request would be valid, False otherwise
        """
        return len(self.get_validation_errors()) == 0

    def get_validation_errors(self) -> List[str]:
        """
        Get a list of validation errors for the current context.

        Returns:
            List of error messages, empty if valid
        """
        errors = []

        # Check required fields
        if not self._model_name:
            errors.append("Model name is required")

        if not self._messages:
            errors.append("At least one message is required")

        # Validate message roles
        if ("validation" in self._schema and
            "message_validation" in self._schema["validation"]):
            validation = self._schema["validation"]["message_validation"]

            if "last_message_role" in validation:
                required_role = validation["last_message_role"]
                if self._messages:
                    last_role = self._messages[-1]["role"]
                    if last_role != required_role:
                        errors.append(f"Last message must be from: {required_role}")

        return errors

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters in the context.

        Returns:
            Dictionary of parameter names to values
        """
        return self._parameters.copy()

    def get_parameter(self, key: str) -> Any:
        """
        Get a parameter value by key.

        Args:
            key: The parameter key

        Returns:
            The parameter value

        Raises:
            ValidationException: If the parameter does not exist
        """
        if key not in self._parameters:
            raise ValidationException(f"Parameter '{key}' not found")
        return self._parameters[key]

    def get_parameter_as(self, key: str, target_type: Type[T], default: Optional[T] = None) -> T:
        """
        Get a parameter value converted to a specific type.

        Args:
            key: The parameter key
            target_type: The type to convert the parameter to
            default: Default value if parameter doesn't exist

        Returns:
            The parameter value converted to the target type

        Raises:
            ValidationException: If the parameter cannot be converted
        """
        if not self.has_parameter(key):
            if default is not None:
                return default
            raise ValidationException(f"Parameter '{key}' not found")

        try:
            value = self._parameters[key]
            if isinstance(value, target_type):
                return value
            return target_type(value)
        except (ValueError, TypeError) as e:
            raise ValidationException(f"Parameter '{key}' cannot be converted to requested type: {str(e)}")

    def has_parameter(self, key: str) -> bool:
        """
        Check if a parameter exists.

        Args:
            key: The parameter key

        Returns:
            True if the parameter exists, False otherwise
        """
        return key in self._parameters

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the context.

        Returns:
            List of message dictionaries
        """
        return self._messages.copy()

    def _resolve_path(self, data: Any, path: List[str]) -> Any:
        """
        Resolve a path in a nested data structure.

        Args:
            data: The data structure to navigate
            path: List of keys/indices to follow

        Returns:
            The value at the specified path

        Raises:
            RuntimeError: If the path cannot be resolved
        """
        current = data

        for key in path:
            if key.isdigit():
                # It's an array index
                index = int(key)
                if not isinstance(current, list) or index >= len(current):
                    raise RuntimeError(f"Invalid array access: index {key}")
                current = current[index]
            else:
                # It's an object key
                if not isinstance(current, dict) or key not in current:
                    raise RuntimeError(f"Invalid object access: key {key}")
                current = current[key]

        return current

    def _parse_json_path(self, path_array: List[Union[str, int]]) -> List[str]:
        """
        Parse a JSON path array into a list of strings.

        Args:
            path_array: Array of path elements

        Returns:
            List of path elements as strings
        """
        path = []
        for element in path_array:
            if isinstance(element, str):
                path.append(element)
            elif isinstance(element, int):
                path.append(str(element))
        return path

    def _validate_message(self, message: Dict[str, Any]) -> None:
        """
        Validate a message object.

        Args:
            message: The message to validate

        Raises:
            ValidationException: If the message is invalid
        """
        if "role" not in message or "content" not in message:
            raise ValidationException("Message must contain 'role' and 'content' fields")

        role = message["role"]
        if self._valid_roles and role not in self._valid_roles:
            raise ValidationException(f"Invalid message role: {role}")

    def _validate_parameter(self, key: str, value: Any) -> None:
        """
        Validate a parameter against the schema.

        Args:
            key: The parameter key
            value: The parameter value

        Raises:
            ValidationException: If the parameter is invalid
        """
        if value is None:
            raise ValidationException(f"Parameter '{key}' cannot be null")

        if "parameters" not in self._schema or key not in self._schema["parameters"]:
            return  # Parameter not defined in schema

        param_def = self._schema["parameters"][key]

        # Add string length validation
        if isinstance(value, str) and "max_length" in param_def:
            max_len = param_def["max_length"]
            if len(value) > max_len:
                raise ValidationException(f"Parameter '{key}' exceeds maximum length of {max_len}")

        # Add enum validation
        if "enum" in param_def:
            if value not in param_def["enum"]:
                raise ValidationException(f"Parameter '{key}' has invalid value")

        # Type validation
        if "type" in param_def:
            expected_type = param_def["type"]
            if expected_type == "integer" and not isinstance(value, int):
                raise ValidationException(f"Parameter '{key}' must be an integer")
            elif expected_type == "float" and not isinstance(value, (int, float)):
                raise ValidationException(f"Parameter '{key}' must be a number")
            elif expected_type == "string" and not isinstance(value, str):
                raise ValidationException(f"Parameter '{key}' must be a string")
            elif expected_type == "boolean" and not isinstance(value, bool):
                raise ValidationException(f"Parameter '{key}' must be a boolean")
            elif expected_type == "array" and not isinstance(value, list):
                raise ValidationException(f"Parameter '{key}' must be an array")

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if "min" in param_def and value < param_def["min"]:
                raise ValidationException(f"Parameter '{key}' must be >= {param_def['min']}")
            if "max" in param_def and value > param_def["max"]:
                raise ValidationException(f"Parameter '{key}' must be <= {param_def['max']}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image data

        Raises:
            RuntimeError: If the image file cannot be read or is too large
        """
        # Check existence and size first
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image file does not exist: {image_path}")

        file_size = path.stat().st_size
        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

        if file_size > MAX_IMAGE_SIZE:
            raise RuntimeError(f"Image file too large: {file_size} bytes")

        try:
            with open(image_path, 'rb') as file:
                image_data = file.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to open image file: {image_path}")

    def _is_base64_encoded(self, data: str) -> bool:
        """
        Check if a string is base64 encoded.

        Args:
            data: The string to check

        Returns:
            True if the string appears to be base64 encoded, False otherwise
        """
        if not data:
            return False

        # Check for data URI scheme (e.g., "data:image/png;base64,...")
        if data.startswith("data:") and ";base64," in data:
            return True

        # Check for valid Base64 characters (ignoring whitespace)
        base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

        padding = 0
        data_len = 0  # Counts non-whitespace chars

        for c in data:
            if c.isspace():
                continue  # Skip whitespace
            if c not in base64_chars:
                return False  # Invalid character
            if c == '=':
                padding += 1
                if padding > 2:
                    return False  # Max 2 padding chars
            data_len += 1

        # Validate length and padding (Base64 length must be divisible by 4)
        return (data_len % 4 == 0) and (padding != 1)  # 1 padding char is invalid
