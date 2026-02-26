"""Unit tests for feature retrieval with cache-aside pattern"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError

from src.common.cache import FeatureCacheClient
from src.common.feature_retrieval import (
    DataSourceError,
    FeatureRetriever,
)


@pytest.fixture
def mock_cache_client():
    """Create a mock cache client"""
    client = AsyncMock(spec=FeatureCacheClient)
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_data_source_client():
    """Create a mock data source client"""
    client = AsyncMock()
    client.get_features = AsyncMock()
    return client


@pytest.fixture
def feature_retriever(mock_cache_client):
    """Create a feature retriever with mock cache client"""
    return FeatureRetriever(
        cache_client=mock_cache_client,
        data_source_client=None,  # Use mock features
        cache_ttl_seconds=300,
        enable_cache=True,
    )


@pytest.mark.asyncio
class TestFeatureRetriever:
    """Test suite for FeatureRetriever"""

    async def test_cache_hit_returns_cached_features(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache hit returns cached features without hitting data source"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        cached_data = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "features": {"age": 30, "account_age_days": 100},
            "cached_at": "2024-01-15T10:00:00",
        }
        mock_cache_client.get.return_value = cached_data

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert result == cached_data["features"]
        mock_cache_client.get.assert_called_once_with(entity_type, entity_id)
        # Cache set should not be called on cache hit
        mock_cache_client.set.assert_not_called()

    async def test_cache_miss_fetches_from_data_source(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache miss triggers data source fetch"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert result is not None
        assert "age" in result  # Mock features should be returned
        mock_cache_client.get.assert_called_once_with(entity_type, entity_id)

    async def test_cache_miss_populates_cache_async(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache is populated asynchronously after data source fetch"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Wait for async cache population
        await asyncio.sleep(0.1)

        # Assert
        assert result is not None
        mock_cache_client.set.assert_called_once()
        call_args = mock_cache_client.set.call_args
        assert call_args[0][0] == entity_type
        assert call_args[0][1] == entity_id
        assert call_args[0][2] == result
        assert call_args[1]["ttl_seconds"] == 300

    async def test_cache_error_falls_back_to_data_source(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache errors trigger fallback to data source"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.side_effect = ConnectionError("Redis connection failed")

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert result is not None
        assert "age" in result  # Mock features should be returned
        mock_cache_client.get.assert_called_once_with(entity_type, entity_id)

    async def test_cache_unavailable_continues_without_cache(
        self, feature_retriever, mock_cache_client
    ):
        """Test that service continues when cache is completely unavailable"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.side_effect = RedisError("Redis unavailable")

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert result is not None
        assert "age" in result  # Mock features should be returned

    async def test_cache_disabled_skips_cache_operations(self, mock_cache_client):
        """Test that cache operations are skipped when cache is disabled"""
        # Arrange
        retriever = FeatureRetriever(
            cache_client=mock_cache_client,
            data_source_client=None,
            enable_cache=False,
        )
        entity_type = "user"
        entity_id = "12345"

        # Act
        result = await retriever.get_features(entity_type, entity_id)

        # Assert
        assert result is not None
        mock_cache_client.get.assert_not_called()
        await asyncio.sleep(0.1)
        mock_cache_client.set.assert_not_called()

    async def test_data_source_error_raises_exception(
        self, mock_cache_client, mock_data_source_client
    ):
        """Test that data source errors are properly raised"""
        # Arrange
        retriever = FeatureRetriever(
            cache_client=mock_cache_client,
            data_source_client=mock_data_source_client,
            enable_cache=True,
        )
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss
        mock_data_source_client.get_features.side_effect = Exception(
            "Database connection failed"
        )

        # Act & Assert
        with pytest.raises(DataSourceError):
            await retriever.get_features(entity_type, entity_id)

    async def test_cache_population_failure_does_not_affect_response(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache population failures don't affect the response"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss
        mock_cache_client.set.side_effect = RedisError("Cache write failed")

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Wait for async cache population attempt
        await asyncio.sleep(0.1)

        # Assert - should still return features despite cache write failure
        assert result is not None
        assert "age" in result

    async def test_invalidate_cache_deletes_cached_entry(
        self, feature_retriever, mock_cache_client
    ):
        """Test cache invalidation"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.delete.return_value = True

        # Act
        result = await feature_retriever.invalidate_cache(entity_type, entity_id)

        # Assert
        assert result is True
        mock_cache_client.delete.assert_called_once_with(entity_type, entity_id)

    async def test_invalidate_cache_when_disabled(self, mock_cache_client):
        """Test that cache invalidation returns False when cache is disabled"""
        # Arrange
        retriever = FeatureRetriever(
            cache_client=mock_cache_client,
            data_source_client=None,
            enable_cache=False,
        )
        entity_type = "user"
        entity_id = "12345"

        # Act
        result = await retriever.invalidate_cache(entity_type, entity_id)

        # Assert
        assert result is False
        mock_cache_client.delete.assert_not_called()

    async def test_invalidate_cache_handles_errors(
        self, feature_retriever, mock_cache_client
    ):
        """Test that cache invalidation handles errors gracefully"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.delete.side_effect = RedisError("Delete failed")

        # Act
        result = await feature_retriever.invalidate_cache(entity_type, entity_id)

        # Assert
        assert result is False

    async def test_get_features_batch_retrieves_multiple_entities(
        self, feature_retriever, mock_cache_client
    ):
        """Test batch feature retrieval"""
        # Arrange
        entity_type = "user"
        entity_ids = ["123", "456", "789"]
        mock_cache_client.get.return_value = None  # All cache misses

        # Act
        result = await feature_retriever.get_features_batch(entity_type, entity_ids)

        # Assert
        assert len(result) == 3
        assert "123" in result
        assert "456" in result
        assert "789" in result
        assert mock_cache_client.get.call_count == 3

    async def test_get_features_batch_handles_partial_failures(
        self, feature_retriever, mock_cache_client
    ):
        """Test that batch retrieval handles partial failures"""
        # Arrange
        entity_type = "user"
        entity_ids = ["123", "456", "789"]
        mock_cache_client.get.return_value = None  # All cache misses

        # Act
        result = await feature_retriever.get_features_batch(entity_type, entity_ids)

        # Assert - all should succeed with mock features
        assert len(result) == 3
        assert "123" in result
        assert "456" in result
        assert "789" in result

    async def test_health_check_returns_component_status(
        self, feature_retriever, mock_cache_client
    ):
        """Test health check returns status of all components"""
        # Arrange
        mock_cache_client.health_check.return_value = True

        # Act
        result = await feature_retriever.health_check()

        # Assert
        assert "cache" in result
        assert result["cache"] is True
        assert "data_source" in result

    async def test_health_check_when_cache_disabled(self, mock_cache_client):
        """Test health check when cache is disabled"""
        # Arrange
        retriever = FeatureRetriever(
            cache_client=mock_cache_client,
            data_source_client=None,
            enable_cache=False,
        )

        # Act
        result = await retriever.health_check()

        # Assert
        assert result["cache"] is None
        mock_cache_client.health_check.assert_not_called()

    async def test_mock_features_for_user_entity(
        self, feature_retriever, mock_cache_client
    ):
        """Test mock feature generation for user entity"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert "age" in result
        assert "account_age_days" in result
        assert "transaction_count_30d" in result
        assert "avg_transaction_amount" in result

    async def test_mock_features_for_transaction_entity(
        self, feature_retriever, mock_cache_client
    ):
        """Test mock feature generation for transaction entity"""
        # Arrange
        entity_type = "transaction"
        entity_id = "txn_123"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act
        result = await feature_retriever.get_features(entity_type, entity_id)

        # Assert
        assert "amount" in result
        assert "merchant_category" in result
        assert "is_international" in result
        assert "time_since_last_transaction_hours" in result

    async def test_custom_cache_ttl(self, mock_cache_client):
        """Test that custom cache TTL is used"""
        # Arrange
        custom_ttl = 600
        retriever = FeatureRetriever(
            cache_client=mock_cache_client,
            data_source_client=None,
            cache_ttl_seconds=custom_ttl,
            enable_cache=True,
        )
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act
        await retriever.get_features(entity_type, entity_id)
        await asyncio.sleep(0.1)  # Wait for async cache population

        # Assert
        call_args = mock_cache_client.set.call_args
        assert call_args[1]["ttl_seconds"] == custom_ttl

    async def test_concurrent_requests_for_same_entity(
        self, feature_retriever, mock_cache_client
    ):
        """Test that concurrent requests for the same entity work correctly"""
        # Arrange
        entity_type = "user"
        entity_id = "12345"
        mock_cache_client.get.return_value = None  # Cache miss

        # Act - make 10 concurrent requests
        tasks = [
            feature_retriever.get_features(entity_type, entity_id) for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 10
        for result in results:
            assert result is not None
            assert "age" in result
