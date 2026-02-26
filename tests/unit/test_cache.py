"""Unit tests for Redis cache client"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from src.common.cache import CacheStats, FeatureCacheClient
from src.common.config import RedisConfig


@pytest.fixture
def redis_config():
    """Create test Redis configuration"""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        password="",
        max_connections=10,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


@pytest.fixture
def cache_client(redis_config):
    """Create cache client instance"""
    return FeatureCacheClient(redis_config)


@pytest.fixture
def mock_redis():
    """Create mock Redis client"""
    mock = AsyncMock()
    mock.ping = AsyncMock()
    mock.config_set = AsyncMock()
    mock.get = AsyncMock()
    mock.setex = AsyncMock()
    mock.delete = AsyncMock()
    mock.close = AsyncMock()
    return mock


class TestCacheStats:
    """Tests for CacheStats class"""

    def test_initial_stats(self):
        """Test initial statistics are zero"""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.errors == 0
        assert stats.total_latency_ms == 0.0
        assert stats.operation_count == 0

    def test_hit_ratio_with_no_operations(self):
        """Test hit ratio is 0 when no operations"""
        stats = CacheStats()
        assert stats.hit_ratio == 0.0

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation"""
        stats = CacheStats()
        stats.hits = 80
        stats.misses = 20
        assert stats.hit_ratio == 0.8

    def test_avg_latency_with_no_operations(self):
        """Test average latency is 0 when no operations"""
        stats = CacheStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation"""
        stats = CacheStats()
        stats.total_latency_ms = 100.0
        stats.operation_count = 10
        assert stats.avg_latency_ms == 10.0

    def test_to_dict(self):
        """Test conversion to dictionary"""
        stats = CacheStats()
        stats.hits = 80
        stats.misses = 20
        stats.errors = 5
        stats.total_latency_ms = 500.0
        stats.operation_count = 100

        result = stats.to_dict()
        assert result["hits"] == 80
        assert result["misses"] == 20
        assert result["errors"] == 5
        assert result["hit_ratio"] == 0.8
        assert result["avg_latency_ms"] == 5.0
        assert result["operation_count"] == 100


class TestFeatureCacheClient:
    """Tests for FeatureCacheClient class"""

    def test_initialization(self, cache_client, redis_config):
        """Test cache client initialization"""
        assert cache_client.config == redis_config
        assert cache_client._initialized is False
        assert cache_client._client is None
        assert cache_client._pool is None

    def test_build_key(self, cache_client):
        """Test cache key building"""
        key = cache_client._build_key("user", "12345")
        assert key == "feature:user:12345"

    @pytest.mark.asyncio
    async def test_initialize_success(self, cache_client, mock_redis):
        """Test successful initialization"""
        with patch("src.common.cache.aioredis.Redis", return_value=mock_redis):
            with patch("src.common.cache.ConnectionPool") as mock_pool:
                await cache_client.initialize()

                assert cache_client._initialized is True
                mock_redis.ping.assert_called_once()
                mock_redis.config_set.assert_called_once_with(
                    "maxmemory-policy", "allkeys-lru"
                )

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, cache_client):
        """Test initialization when already initialized"""
        cache_client._initialized = True
        await cache_client.initialize()
        # Should return early without doing anything

    @pytest.mark.asyncio
    async def test_close(self, cache_client, mock_redis):
        """Test closing connection"""
        mock_pool = AsyncMock()
        cache_client._client = mock_redis
        cache_client._pool = mock_pool
        cache_client._initialized = True

        await cache_client.close()

        mock_redis.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert cache_client._initialized is False

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_client, mock_redis):
        """Test successful cache hit"""
        features = {"age": 30, "income": 50000}
        cache_value = {
            "entity_id": "12345",
            "entity_type": "user",
            "features": features,
            "cached_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(cache_value)

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.get("user", "12345")

        assert result == cache_value
        assert cache_client._stats.hits == 1
        assert cache_client._stats.misses == 0
        assert cache_client._stats.operation_count == 1
        mock_redis.get.assert_called_once_with("feature:user:12345")

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_client, mock_redis):
        """Test cache miss"""
        mock_redis.get.return_value = None

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.get("user", "12345")

        assert result is None
        assert cache_client._stats.hits == 0
        assert cache_client._stats.misses == 1
        assert cache_client._stats.operation_count == 1

    @pytest.mark.asyncio
    async def test_get_json_decode_error(self, cache_client, mock_redis):
        """Test handling of JSON decode error"""
        mock_redis.get.return_value = "invalid json"

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.get("user", "12345")

        assert result is None
        assert cache_client._stats.errors == 1

    @pytest.mark.asyncio
    async def test_get_connection_error(self, cache_client, mock_redis):
        """Test handling of connection error"""
        mock_redis.get.side_effect = ConnectionError("Connection failed")

        cache_client._client = mock_redis
        cache_client._initialized = True

        # Should retry and eventually raise
        with pytest.raises(ConnectionError):
            await cache_client.get("user", "12345")

        assert cache_client._stats.errors > 0

    @pytest.mark.asyncio
    async def test_set_success(self, cache_client, mock_redis):
        """Test successful cache set"""
        features = {"age": 30, "income": 50000}
        mock_redis.setex.return_value = True

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.set("user", "12345", features, ttl_seconds=300)

        assert result is True
        assert cache_client._stats.operation_count == 1
        mock_redis.setex.assert_called_once()

        # Verify the call arguments
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "feature:user:12345"  # key
        assert call_args[0][1] == 300  # ttl
        # Verify JSON structure
        cached_data = json.loads(call_args[0][2])
        assert cached_data["entity_id"] == "12345"
        assert cached_data["entity_type"] == "user"
        assert cached_data["features"] == features

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache_client, mock_redis):
        """Test cache set with custom TTL"""
        features = {"age": 30}
        mock_redis.setex.return_value = True

        cache_client._client = mock_redis
        cache_client._initialized = True

        await cache_client.set("user", "12345", features, ttl_seconds=600)

        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 600  # ttl

    @pytest.mark.asyncio
    async def test_set_json_encode_error(self, cache_client, mock_redis):
        """Test handling of JSON encode error"""
        # Create an object that can't be JSON serialized
        features = {"func": lambda x: x}

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.set("user", "12345", features)

        assert result is False
        assert cache_client._stats.errors == 1

    @pytest.mark.asyncio
    async def test_set_connection_error(self, cache_client, mock_redis):
        """Test handling of connection error during set"""
        features = {"age": 30}
        mock_redis.setex.side_effect = ConnectionError("Connection failed")

        cache_client._client = mock_redis
        cache_client._initialized = True

        # Should retry and eventually raise
        with pytest.raises(ConnectionError):
            await cache_client.set("user", "12345", features)

        assert cache_client._stats.errors > 0

    @pytest.mark.asyncio
    async def test_delete_success(self, cache_client, mock_redis):
        """Test successful cache delete"""
        mock_redis.delete.return_value = 1  # Key was deleted

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.delete("user", "12345")

        assert result is True
        assert cache_client._stats.operation_count == 1
        mock_redis.delete.assert_called_once_with("feature:user:12345")

    @pytest.mark.asyncio
    async def test_delete_key_not_found(self, cache_client, mock_redis):
        """Test delete when key doesn't exist"""
        mock_redis.delete.return_value = 0  # Key didn't exist

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.delete("user", "12345")

        assert result is False
        assert cache_client._stats.operation_count == 1

    @pytest.mark.asyncio
    async def test_delete_connection_error(self, cache_client, mock_redis):
        """Test handling of connection error during delete"""
        mock_redis.delete.side_effect = ConnectionError("Connection failed")

        cache_client._client = mock_redis
        cache_client._initialized = True

        # Should retry and eventually raise
        with pytest.raises(ConnectionError):
            await cache_client.delete("user", "12345")

        assert cache_client._stats.errors > 0

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_client):
        """Test getting cache statistics"""
        cache_client._stats.hits = 100
        cache_client._stats.misses = 20

        stats = await cache_client.get_stats()

        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.hit_ratio == 100 / 120

    @pytest.mark.asyncio
    async def test_reset_stats(self, cache_client):
        """Test resetting cache statistics"""
        cache_client._stats.hits = 100
        cache_client._stats.misses = 20

        await cache_client.reset_stats()

        assert cache_client._stats.hits == 0
        assert cache_client._stats.misses == 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, cache_client, mock_redis):
        """Test successful health check"""
        mock_redis.ping.return_value = True

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.health_check()

        assert result is True
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, cache_client, mock_redis):
        """Test failed health check"""
        mock_redis.ping.side_effect = ConnectionError("Connection failed")

        cache_client._client = mock_redis
        cache_client._initialized = True

        result = await cache_client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_flush_all(self, cache_client, mock_redis):
        """Test flushing all cache keys"""
        mock_redis.flushdb.return_value = True

        cache_client._client = mock_redis
        cache_client._initialized = True

        await cache_client.flush_all()

        mock_redis.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_all_error(self, cache_client, mock_redis):
        """Test error during flush all"""
        mock_redis.flushdb.side_effect = RedisError("Flush failed")

        cache_client._client = mock_redis
        cache_client._initialized = True

        with pytest.raises(RedisError):
            await cache_client.flush_all()

    @pytest.mark.asyncio
    async def test_auto_initialize_on_get(self, cache_client, mock_redis):
        """Test automatic initialization on first get"""
        mock_redis.get.return_value = None

        with patch("src.common.cache.aioredis.Redis", return_value=mock_redis):
            with patch("src.common.cache.ConnectionPool"):
                result = await cache_client.get("user", "12345")

                assert cache_client._initialized is True
                assert result is None

    @pytest.mark.asyncio
    async def test_auto_initialize_on_set(self, cache_client, mock_redis):
        """Test automatic initialization on first set"""
        mock_redis.setex.return_value = True

        with patch("src.common.cache.aioredis.Redis", return_value=mock_redis):
            with patch("src.common.cache.ConnectionPool"):
                result = await cache_client.set("user", "12345", {"age": 30})

                assert cache_client._initialized is True
                assert result is True

    @pytest.mark.asyncio
    async def test_auto_initialize_on_delete(self, cache_client, mock_redis):
        """Test automatic initialization on first delete"""
        mock_redis.delete.return_value = 1

        with patch("src.common.cache.aioredis.Redis", return_value=mock_redis):
            with patch("src.common.cache.ConnectionPool"):
                result = await cache_client.delete("user", "12345")

                assert cache_client._initialized is True
                assert result is True

    @pytest.mark.asyncio
    async def test_latency_tracking(self, cache_client, mock_redis):
        """Test that latency is tracked for operations"""
        # Use a real async function that takes time
        async def delayed_get(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms delay to ensure measurable latency
            return None

        mock_redis.get = delayed_get

        cache_client._client = mock_redis
        cache_client._initialized = True

        await cache_client.get("user", "12345")

        assert cache_client._stats.operation_count == 1
        # With 10ms sleep, latency should be at least 5ms (accounting for overhead)
        assert cache_client._stats.total_latency_ms >= 5.0
        assert cache_client._stats.avg_latency_ms >= 5.0

    @pytest.mark.asyncio
    async def test_multiple_operations_stats(self, cache_client, mock_redis):
        """Test statistics tracking across multiple operations"""
        mock_redis.get.return_value = json.dumps({"features": {"age": 30}})
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = 1

        cache_client._client = mock_redis
        cache_client._initialized = True

        # Perform multiple operations
        await cache_client.get("user", "1")  # hit
        await cache_client.get("user", "2")  # hit
        mock_redis.get.return_value = None
        await cache_client.get("user", "3")  # miss
        await cache_client.set("user", "4", {"age": 25})
        await cache_client.delete("user", "5")

        assert cache_client._stats.hits == 2
        assert cache_client._stats.misses == 1
        assert cache_client._stats.operation_count == 5
