"""Redis cache client with async support for feature caching"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from redis.exceptions import ConnectionError, RedisError, TimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import RedisConfig

logger = logging.getLogger(__name__)


class CacheStats:
    """Cache statistics tracking"""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.errors: int = 0
        self.total_latency_ms: float = 0.0
        self.operation_count: int = 0

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average operation latency"""
        return (
            self.total_latency_ms / self.operation_count
            if self.operation_count > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_ratio": self.hit_ratio,
            "avg_latency_ms": self.avg_latency_ms,
            "operation_count": self.operation_count,
        }


class FeatureCacheClient:
    """
    Redis-based cache client for feature storage with async support.

    Features:
    - Async operations for high performance
    - Connection pooling for efficient resource usage
    - Automatic retry logic with exponential backoff
    - TTL-based expiration
    - LRU eviction policy
    - Cache statistics tracking (hits, misses, latency)
    """

    def __init__(self, config: RedisConfig):
        """
        Initialize the cache client.

        Args:
            config: Redis configuration
        """
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._stats = CacheStats()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection pool and client"""
        if self._initialized:
            return

        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password if self.config.password else None,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,  # Automatically decode responses to strings
            )

            # Create Redis client
            self._client = aioredis.Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()

            # Configure LRU eviction policy
            await self._client.config_set("maxmemory-policy", "allkeys-lru")

            self._initialized = True
            logger.info(
                f"Redis cache client initialized: {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache client: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection pool"""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._initialized = False
        logger.info("Redis cache client closed")

    def _build_key(self, entity_type: str, entity_id: str) -> str:
        """
        Build cache key from entity type and ID.

        Args:
            entity_type: Type of entity (e.g., 'user', 'transaction')
            entity_id: Entity identifier

        Returns:
            Cache key string
        """
        return f"feature:{entity_type}:{entity_id}"

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        reraise=True,
    )
    async def get(
        self, entity_type: str, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from cache.

        Args:
            entity_type: Type of entity (e.g., 'user', 'transaction')
            entity_id: Entity identifier

        Returns:
            Feature dictionary if found, None otherwise

        Raises:
            RedisError: If cache operation fails after retries
        """
        if not self._initialized:
            await self.initialize()

        key = self._build_key(entity_type, entity_id)
        start_time = asyncio.get_event_loop().time()

        try:
            value = await self._client.get(key)

            # Track latency
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._stats.total_latency_ms += latency_ms
            self._stats.operation_count += 1

            if value is None:
                self._stats.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None

            # Deserialize JSON
            features = json.loads(value)
            self._stats.hits += 1
            logger.debug(f"Cache hit: {key} (latency: {latency_ms:.2f}ms)")
            return features

        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.warning(f"Cache connection error for key {key}: {e}")
            raise

        except json.JSONDecodeError as e:
            self._stats.errors += 1
            logger.error(f"Failed to deserialize cached value for key {key}: {e}")
            return None

        except RedisError as e:
            self._stats.errors += 1
            logger.error(f"Redis error for key {key}: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        reraise=True,
    )
    async def set(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        ttl_seconds: int = 300,
    ) -> bool:
        """
        Store features in cache with TTL.

        Args:
            entity_type: Type of entity (e.g., 'user', 'transaction')
            entity_id: Entity identifier
            features: Feature dictionary to cache
            ttl_seconds: Time-to-live in seconds (default: 300)

        Returns:
            True if successful, False otherwise

        Raises:
            RedisError: If cache operation fails after retries
        """
        if not self._initialized:
            await self.initialize()

        key = self._build_key(entity_type, entity_id)
        start_time = asyncio.get_event_loop().time()

        try:
            # Add metadata
            cache_value = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "features": features,
                "cached_at": datetime.utcnow().isoformat(),
            }

            # Serialize to JSON
            value = json.dumps(cache_value)

            # Store with TTL
            await self._client.setex(key, ttl_seconds, value)

            # Track latency
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._stats.total_latency_ms += latency_ms
            self._stats.operation_count += 1

            logger.debug(
                f"Cache set: {key} (ttl: {ttl_seconds}s, latency: {latency_ms:.2f}ms)"
            )
            return True

        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.warning(f"Cache connection error for key {key}: {e}")
            raise

        except TypeError as e:
            self._stats.errors += 1
            logger.error(f"Failed to serialize features for key {key}: {e}")
            return False

        except RedisError as e:
            self._stats.errors += 1
            logger.error(f"Redis error for key {key}: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        reraise=True,
    )
    async def delete(self, entity_type: str, entity_id: str) -> bool:
        """
        Delete features from cache.

        Args:
            entity_type: Type of entity (e.g., 'user', 'transaction')
            entity_id: Entity identifier

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            RedisError: If cache operation fails after retries
        """
        if not self._initialized:
            await self.initialize()

        key = self._build_key(entity_type, entity_id)
        start_time = asyncio.get_event_loop().time()

        try:
            result = await self._client.delete(key)

            # Track latency
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._stats.total_latency_ms += latency_ms
            self._stats.operation_count += 1

            deleted = result > 0
            logger.debug(
                f"Cache delete: {key} (deleted: {deleted}, latency: {latency_ms:.2f}ms)"
            )
            return deleted

        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.warning(f"Cache connection error for key {key}: {e}")
            raise

        except RedisError as e:
            self._stats.errors += 1
            logger.error(f"Redis error for key {key}: {e}")
            raise

    async def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object with hit/miss/latency metrics
        """
        return self._stats

    async def reset_stats(self) -> None:
        """Reset cache statistics"""
        self._stats = CacheStats()
        logger.info("Cache statistics reset")

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False

    async def flush_all(self) -> None:
        """
        Flush all keys from the current database.

        Warning: This will delete all cached data!
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self._client.flushdb()
            logger.warning("All cache keys flushed")
        except RedisError as e:
            logger.error(f"Failed to flush cache: {e}")
            raise
