"""Feature retrieval with cache-aside pattern and fallback logic"""

import asyncio
import logging
from typing import Any, Dict, Optional

from redis.exceptions import RedisError

from .cache import FeatureCacheClient

logger = logging.getLogger(__name__)


class FeatureRetrievalError(Exception):
    """Base exception for feature retrieval errors"""

    pass


class DataSourceError(FeatureRetrievalError):
    """Exception raised when primary data source fails"""

    pass


class FeatureRetriever:
    """
    Feature retrieval service implementing cache-aside pattern.

    The cache-aside pattern works as follows:
    1. Check cache first (cache-first strategy)
    2. On cache miss, fetch from primary data source
    3. Asynchronously populate cache after DB fetch
    4. Handle cache unavailability gracefully (continue without cache)

    This ensures low-latency feature access while maintaining resilience.
    """

    def __init__(
        self,
        cache_client: FeatureCacheClient,
        data_source_client: Optional[Any] = None,
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
    ):
        """
        Initialize feature retriever.

        Args:
            cache_client: Redis cache client for feature caching
            data_source_client: Primary data source client (e.g., PostgreSQL)
            cache_ttl_seconds: TTL for cached features (default: 300s)
            enable_cache: Whether to use cache (default: True)
        """
        self.cache_client = cache_client
        self.data_source_client = data_source_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_cache = enable_cache

    async def get_features(
        self, entity_type: str, entity_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve features with cache-aside pattern.

        Flow:
        1. Try cache first if enabled
        2. On cache miss or cache error, fetch from data source
        3. Asynchronously populate cache after DB fetch
        4. Return features to caller without blocking on cache update

        Args:
            entity_type: Type of entity (e.g., 'user', 'transaction')
            entity_id: Entity identifier

        Returns:
            Feature dictionary

        Raises:
            FeatureRetrievalError: If both cache and data source fail
        """
        # Step 1: Try cache first (if enabled)
        if self.enable_cache:
            try:
                cached_features = await self._get_from_cache(entity_type, entity_id)
                if cached_features is not None:
                    logger.debug(
                        f"Cache hit for {entity_type}:{entity_id}",
                        extra={"entity_type": entity_type, "entity_id": entity_id},
                    )
                    return cached_features["features"]
                else:
                    logger.debug(
                        f"Cache miss for {entity_type}:{entity_id}",
                        extra={"entity_type": entity_type, "entity_id": entity_id},
                    )
            except RedisError as e:
                # Cache error - log warning and continue to data source
                logger.warning(
                    f"Cache error for {entity_type}:{entity_id}, falling back to data source: {e}",
                    extra={
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "error": str(e),
                    },
                )

        # Step 2: Fetch from primary data source
        try:
            features = await self._get_from_data_source(entity_type, entity_id)
        except Exception as e:
            logger.error(
                f"Failed to retrieve features from data source for {entity_type}:{entity_id}: {e}",
                extra={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "error": str(e),
                },
            )
            raise DataSourceError(
                f"Failed to retrieve features from data source: {e}"
            ) from e

        # Step 3: Asynchronously populate cache (don't block on this)
        if self.enable_cache:
            asyncio.create_task(
                self._populate_cache_async(entity_type, entity_id, features)
            )

        return features

    async def _get_from_cache(
        self, entity_type: str, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from cache.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Cached feature dictionary or None if not found

        Raises:
            RedisError: If cache operation fails
        """
        return await self.cache_client.get(entity_type, entity_id)

    async def _get_from_data_source(
        self, entity_type: str, entity_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve features from primary data source.

        This is a placeholder implementation. In production, this would
        query PostgreSQL, DynamoDB, or another primary data store.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Feature dictionary from data source

        Raises:
            DataSourceError: If data source query fails
        """
        if self.data_source_client is None:
            # Placeholder: return mock features for testing
            logger.warning(
                f"No data source client configured, returning mock features for {entity_type}:{entity_id}"
            )
            return self._get_mock_features(entity_type, entity_id)

        # In production, this would be:
        # return await self.data_source_client.get_features(entity_type, entity_id)
        raise NotImplementedError("Data source client integration not yet implemented")

    async def _populate_cache_async(
        self, entity_type: str, entity_id: str, features: Dict[str, Any]
    ) -> None:
        """
        Asynchronously populate cache after DB fetch.

        This runs in the background and doesn't block the response.
        If cache population fails, we log the error but don't raise.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            features: Feature dictionary to cache
        """
        try:
            await self.cache_client.set(
                entity_type, entity_id, features, ttl_seconds=self.cache_ttl_seconds
            )
            logger.debug(
                f"Cache populated for {entity_type}:{entity_id}",
                extra={"entity_type": entity_type, "entity_id": entity_id},
            )
        except RedisError as e:
            # Log error but don't raise - cache population is best-effort
            logger.warning(
                f"Failed to populate cache for {entity_type}:{entity_id}: {e}",
                extra={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "error": str(e),
                },
            )
        except Exception as e:
            logger.error(
                f"Unexpected error populating cache for {entity_type}:{entity_id}: {e}",
                extra={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "error": str(e),
                },
            )

    def _get_mock_features(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """
        Generate mock features for testing.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Mock feature dictionary
        """
        if entity_type == "user":
            return {
                "age": 32,
                "account_age_days": 450,
                "transaction_count_30d": 15,
                "avg_transaction_amount": 125.50,
            }
        elif entity_type == "transaction":
            return {
                "amount": 99.99,
                "merchant_category": "retail",
                "is_international": False,
                "time_since_last_transaction_hours": 24,
            }
        else:
            return {"entity_id": entity_id, "entity_type": entity_type}

    async def invalidate_cache(self, entity_type: str, entity_id: str) -> bool:
        """
        Invalidate cached features for an entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            True if cache was invalidated, False otherwise
        """
        if not self.enable_cache:
            return False

        try:
            deleted = await self.cache_client.delete(entity_type, entity_id)
            if deleted:
                logger.info(
                    f"Cache invalidated for {entity_type}:{entity_id}",
                    extra={"entity_type": entity_type, "entity_id": entity_id},
                )
            return deleted
        except RedisError as e:
            logger.warning(
                f"Failed to invalidate cache for {entity_type}:{entity_id}: {e}",
                extra={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "error": str(e),
                },
            )
            return False

    async def get_features_batch(
        self, entity_type: str, entity_ids: list[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for multiple entities.

        This method fetches features for multiple entities in parallel,
        using cache-aside pattern for each entity.

        Args:
            entity_type: Type of entity
            entity_ids: List of entity identifiers

        Returns:
            Dictionary mapping entity_id to features
        """
        tasks = [
            self.get_features(entity_type, entity_id) for entity_id in entity_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        features_map = {}
        for entity_id, result in zip(entity_ids, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to retrieve features for {entity_type}:{entity_id}: {result}",
                    extra={
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "error": str(result),
                    },
                )
            else:
                features_map[entity_id] = result

        return features_map

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of cache and data source.

        Returns:
            Dictionary with health status of each component
        """
        health = {}

        # Check cache health
        if self.enable_cache:
            health["cache"] = await self.cache_client.health_check()
        else:
            health["cache"] = None

        # Check data source health (if configured)
        if self.data_source_client is not None:
            # In production, this would check data source connectivity
            health["data_source"] = True
        else:
            health["data_source"] = None

        return health
