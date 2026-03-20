"""
Schema Registry and Context Factory for LLM API Management

This module provides utilities for managing multiple provider schemas and
creating contexts efficiently with caching support.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from functools import lru_cache
import logging

from general_context import GeneralContext, ContextConfig, SchemaException

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """
    Immutable configuration for schema paths.

    Thread-safe after construction. Create once and share across threads.

    Example:
        registry = (SchemaRegistry.builder()
                    .set_schema_directory("./schemas")
                    .register_schema("custom_claude", "/path/to/custom/claude.json")
                    .build())
    """

    class Builder:
        """Builder for SchemaRegistry with fluent interface."""

        def __init__(self):
            self._schema_directory = Path("./schemas")
            self._provider_paths: Dict[str, Path] = {}

        def set_schema_directory(self, directory: str) -> 'SchemaRegistry.Builder':
            """Set the default directory for schema files."""
            self._schema_directory = Path(directory)
            return self

        def register_schema(self, provider_name: str, schema_path: str) -> 'SchemaRegistry.Builder':
            """Register a specific schema path for a provider."""
            if not provider_name:
                raise ValueError("Provider name cannot be empty")
            self._provider_paths[provider_name] = Path(schema_path)
            return self

        def register_schemas(self, schemas: Dict[str, str]) -> 'SchemaRegistry.Builder':
            """Register multiple schemas at once."""
            for name, path in schemas.items():
                self.register_schema(name, path)
            return self

        def build(self) -> 'SchemaRegistry':
            """Build the immutable SchemaRegistry."""
            return SchemaRegistry(self._schema_directory, self._provider_paths.copy())

    def __init__(self, schema_directory: Path, provider_paths: Dict[str, Path]):
        """Private constructor - use builder() instead."""
        self._schema_directory = schema_directory
        self._provider_paths = provider_paths
        # Make immutable
        self._initialized = True

    def __setattr__(self, name, value):
        """Prevent modification after initialization."""
        if hasattr(self, '_initialized') and self._initialized:
            raise AttributeError(f"SchemaRegistry is immutable")
        super().__setattr__(name, value)

    @classmethod
    def builder(cls) -> Builder:
        """Create a new builder for SchemaRegistry."""
        return cls.Builder()

    def resolve_schema_path(self, provider_name: str) -> Path:
        """
        Resolve the schema path for a given provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Absolute path to the schema file

        Raises:
            ValueError: If provider name is empty
        """
        if not provider_name:
            raise ValueError("Provider name cannot be empty")

        # Check registered paths first
        if provider_name in self._provider_paths:
            return self._provider_paths[provider_name].resolve()

        # Fall back to schema directory
        return (self._schema_directory / f"{provider_name}.json").resolve()

    def get_available_providers(self) -> List[str]:
        """
        Get list of all available providers.

        Returns:
            Sorted list of provider names
        """
        providers: Set[str] = set()

        # From registered paths
        for name, path in self._provider_paths.items():
            if path.exists():
                providers.add(name)

        # From schema directory
        if self._schema_directory.exists():
            for entry in self._schema_directory.iterdir():
                if entry.is_file() and entry.suffix == '.json':
                    providers.add(entry.stem)

        return sorted(providers)

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider's schema is available."""
        return self.resolve_schema_path(provider_name).exists()


@dataclass
class CacheStats:
    """Statistics for schema cache."""
    cache_size: int
    hit_count: int
    miss_count: int

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a value between 0.0 and 1.0."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class ContextFactory:
    """
    Factory for creating GeneralContext instances with schema caching.

    Thread-safe. Can be shared across threads.

    Example:
        registry = SchemaRegistry.builder().build()
        factory = ContextFactory(registry)

        # Create new context
        context = factory.create_context("claude")

        # Or use thread-local context
        context = factory.get_thread_local_context("claude")
    """

    def __init__(self, registry: SchemaRegistry, cache_size: Optional[int] = 128):
        """
        Initialize the context factory.

        Args:
            registry: Schema registry to use
            cache_size: Maximum number of schemas to cache (None for unlimited)
        """
        if not registry:
            raise ValueError("Registry cannot be None")

        self._registry = registry
        self._cache_lock = threading.RLock()
        self._cache_stats_lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

        # Use LRU cache for schema storage
        if cache_size:
            self._load_schema_cached = lru_cache(maxsize=cache_size)(self._load_schema)
        else:
            self._load_schema_cached = lru_cache(maxsize=None)(self._load_schema)

        # Thread-local storage for contexts
        self._thread_local = threading.local()

    def create_context(self, provider_name: str,
                      config: Optional[ContextConfig] = None) -> GeneralContext:
        """
        Create a new context instance.

        Each call creates a new independent instance suitable for thread-local use.

        Args:
            provider_name: Name of the provider
            config: Optional context configuration

        Returns:
            New GeneralContext instance

        Raises:
            SchemaException: If schema file not found or invalid
        """
        schema_path = self._registry.resolve_schema_path(provider_name)

        if not schema_path.exists():
            raise SchemaException(
                f"Schema file not found for provider: {provider_name} at {schema_path}"
            )

        # Load schema (with caching)
        schema = self._get_cached_schema(str(schema_path))

        # Create new context with the schema
        return GeneralContext(schema, config or ContextConfig())

    def get_thread_local_context(self, provider_name: str,
                                config: Optional[ContextConfig] = None) -> GeneralContext:
        """
        Get or create a thread-local context.

        The context is created on first access per thread.

        Args:
            provider_name: Name of the provider
            config: Optional context configuration (used only on first creation)

        Returns:
            Thread-local GeneralContext instance
        """
        if not hasattr(self._thread_local, 'contexts'):
            self._thread_local.contexts = {}

        if provider_name not in self._thread_local.contexts:
            self._thread_local.contexts[provider_name] = self.create_context(provider_name, config)

        return self._thread_local.contexts[provider_name]

    def clear_cache(self) -> None:
        """Clear the schema cache. Useful for development/testing when schemas change."""
        with self._cache_lock:
            self._load_schema_cached.cache_clear()
            logger.info("Schema cache cleared")

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._cache_stats_lock:
            cache_info = self._load_schema_cached.cache_info()
            return CacheStats(
                cache_size=cache_info.currsize,
                hit_count=cache_info.hits,
                miss_count=cache_info.misses
            )

    def _get_cached_schema(self, schema_path: str) -> Dict[str, Any]:
        """Get schema with cache tracking."""
        return self._load_schema_cached(schema_path)

    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load schema from a local file path or remote URL (cached by LRU decorator).

        Remote URLs (http:// or https://) enable centralized schema distribution —
        teams can host provider configs on an internal server and all clients
        pick up updates automatically without redeployment.
        """
        # Support HTTP/HTTPS URLs for remote schema loading
        if schema_path.startswith(('http://', 'https://')):
            import urllib.request
            with urllib.request.urlopen(schema_path) as response:
                content = response.read()
            schema = json.loads(content)
            logger.debug(f"Loaded remote schema from {schema_path}")
            return schema

        path = Path(schema_path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                logger.debug(f"Loaded schema from {path}")
                return schema
        except FileNotFoundError:
            raise SchemaException(f"Schema file not found: {path}")
        except json.JSONDecodeError as e:
            raise SchemaException(f"Failed to parse schema JSON at {path}: {e}")


class ProviderContext:
    """
    Convenience wrapper for a specific provider.

    Simplifies working with a single provider by managing the context lifecycle.

    Example:
        # Create provider-specific context
        claude = ProviderContext(factory, "claude")

        # Use it
        claude.get().add_user_message("Hello")
        response = claude.get().build_request()

        # Reset when needed
        claude.reset()
    """

    def __init__(self, factory: ContextFactory, provider_name: str,
                 config: Optional[ContextConfig] = None):
        """
        Initialize provider context.

        Args:
            factory: Context factory to use
            provider_name: Name of the provider
            config: Optional context configuration
        """
        self._factory = factory
        self._provider_name = provider_name
        self._config = config or ContextConfig()
        self._thread_local = threading.local()

    def get(self) -> GeneralContext:
        """
        Get the thread-local context for this provider.

        Creates the context on first access per thread.

        Returns:
            Thread-local GeneralContext instance
        """
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = self._factory.create_context(
                self._provider_name, self._config
            )
        return self._thread_local.context

    def reset(self) -> None:
        """Reset the context to its initial state."""
        self.get().reset()

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name


# Convenience functions for common use cases

def create_registry_from_directory(directory: str = "./schemas") -> SchemaRegistry:
    """
    Create a schema registry from a directory of schema files.

    Args:
        directory: Path to directory containing schema files

    Returns:
        Configured SchemaRegistry
    """
    return SchemaRegistry.builder().set_schema_directory(directory).build()


def create_factory_with_defaults(schema_dir: str = "./schemas",
                               cache_size: Optional[int] = 128) -> ContextFactory:
    """
    Create a context factory with default settings.

    Args:
        schema_dir: Directory containing schema files
        cache_size: Maximum number of schemas to cache

    Returns:
        Configured ContextFactory
    """
    registry = create_registry_from_directory(schema_dir)
    return ContextFactory(registry, cache_size)
