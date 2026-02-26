"""
Demo script showing cache-aside pattern with fallback.

This demonstrates:
1. Cache-first strategy (check cache before data source)
2. Fallback to data source on cache miss
3. Async cache population after DB fetch
4. Graceful handling of cache unavailability
"""

import asyncio
import logging

from src.common.cache import FeatureCacheClient
from src.common.config import RedisConfig
from src.common.feature_retrieval import FeatureRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_cache_aside_pattern():
    """Demonstrate cache-aside pattern with various scenarios"""

    # Initialize cache client
    redis_config = RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        max_connections=10,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
    )

    cache_client = FeatureCacheClient(redis_config)

    try:
        await cache_client.initialize()
        logger.info("✓ Cache client initialized")
    except Exception as e:
        logger.warning(f"✗ Cache unavailable: {e}")
        logger.info("Continuing without cache (graceful degradation)")

    # Initialize feature retriever
    retriever = FeatureRetriever(
        cache_client=cache_client,
        data_source_client=None,  # Use mock features
        cache_ttl_seconds=300,
        enable_cache=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 1: Cache Miss → Data Source → Cache Population")
    logger.info("=" * 60)

    # First request - cache miss
    entity_type = "user"
    entity_id = "12345"

    logger.info(f"Fetching features for {entity_type}:{entity_id} (first time)")
    features = await retriever.get_features(entity_type, entity_id)
    logger.info(f"✓ Features retrieved: {features}")

    # Wait for async cache population
    await asyncio.sleep(0.2)

    # Get cache stats
    stats = await cache_client.get_stats()
    logger.info(f"Cache stats: {stats.to_dict()}")

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 2: Cache Hit (Fast Path)")
    logger.info("=" * 60)

    # Second request - cache hit
    logger.info(f"Fetching features for {entity_type}:{entity_id} (second time)")
    features = await retriever.get_features(entity_type, entity_id)
    logger.info(f"✓ Features retrieved from cache: {features}")

    # Get updated cache stats
    stats = await cache_client.get_stats()
    logger.info(f"Cache stats: {stats.to_dict()}")
    logger.info(f"Cache hit ratio: {stats.hit_ratio:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 3: Batch Feature Retrieval")
    logger.info("=" * 60)

    # Batch retrieval
    entity_ids = ["user_1", "user_2", "user_3"]
    logger.info(f"Fetching features for {len(entity_ids)} users in parallel")
    features_map = await retriever.get_features_batch(entity_type, entity_ids)
    logger.info(f"✓ Retrieved features for {len(features_map)} users")

    for entity_id, features in features_map.items():
        logger.info(f"  - {entity_id}: {list(features.keys())}")

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 4: Cache Invalidation")
    logger.info("=" * 60)

    # Invalidate cache
    logger.info(f"Invalidating cache for user:12345")
    invalidated = await retriever.invalidate_cache("user", "12345")
    logger.info(f"✓ Cache invalidated: {invalidated}")

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 5: Different Entity Types")
    logger.info("=" * 60)

    # Transaction features
    logger.info("Fetching transaction features")
    txn_features = await retriever.get_features("transaction", "txn_123")
    logger.info(f"✓ Transaction features: {txn_features}")

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO 6: Health Check")
    logger.info("=" * 60)

    # Health check
    health = await retriever.health_check()
    logger.info(f"System health: {health}")

    # Final cache stats
    logger.info("\n" + "=" * 60)
    logger.info("FINAL CACHE STATISTICS")
    logger.info("=" * 60)

    stats = await cache_client.get_stats()
    logger.info(f"Total operations: {stats.operation_count}")
    logger.info(f"Cache hits: {stats.hits}")
    logger.info(f"Cache misses: {stats.misses}")
    logger.info(f"Cache errors: {stats.errors}")
    logger.info(f"Hit ratio: {stats.hit_ratio:.2%}")
    logger.info(f"Average latency: {stats.avg_latency_ms:.2f}ms")

    # Cleanup
    await cache_client.close()
    logger.info("\n✓ Demo completed successfully")


async def demo_cache_unavailable():
    """Demonstrate graceful degradation when cache is unavailable"""

    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Graceful Degradation (Cache Unavailable)")
    logger.info("=" * 60)

    # Configure with invalid Redis connection
    redis_config = RedisConfig(
        host="invalid-host",  # This will fail to connect
        port=6379,
        db=0,
        max_connections=10,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
    )

    cache_client = FeatureCacheClient(redis_config)

    # Initialize feature retriever (cache will be unavailable)
    retriever = FeatureRetriever(
        cache_client=cache_client,
        data_source_client=None,
        cache_ttl_seconds=300,
        enable_cache=True,
    )

    try:
        logger.info("Attempting to fetch features with unavailable cache...")
        features = await retriever.get_features("user", "12345")
        logger.info(f"✓ Features retrieved despite cache unavailability: {features}")
        logger.info("✓ Service continued without cache (graceful degradation)")
    except Exception as e:
        logger.error(f"✗ Failed to retrieve features: {e}")


if __name__ == "__main__":
    logger.info("Starting Cache-Aside Pattern Demo")
    logger.info("=" * 60)

    # Run main demo
    asyncio.run(demo_cache_aside_pattern())

    # Run graceful degradation demo
    asyncio.run(demo_cache_unavailable())
