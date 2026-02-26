"""Tests for configuration management"""

import os
from pathlib import Path

import pytest

from src.common.config import Config, load_config


def test_load_development_config() -> None:
    """Test loading development configuration"""
    config = load_config("development")

    assert config.environment == "development"
    assert config.inference_service.port == 8000
    assert config.redis.host == "localhost"
    assert config.kafka.bootstrap_servers == ["localhost:9092"]


def test_load_staging_config() -> None:
    """Test loading staging configuration"""
    config = load_config("staging")

    assert config.environment == "staging"
    assert config.inference_service.workers == 4
    assert config.redis.host == "redis-cluster"


def test_load_production_config() -> None:
    """Test loading production configuration"""
    config = load_config("production")

    assert config.environment == "production"
    assert config.inference_service.workers == 8
    assert config.redis.max_connections == 200


def test_load_config_with_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test configuration override with environment variables"""
    monkeypatch.setenv("REDIS_HOST", "custom-redis")
    monkeypatch.setenv("REDIS_PORT", "6380")

    config = load_config("development")

    assert config.redis.host == "custom-redis"
    assert config.redis.port == 6380


def test_load_config_missing_file() -> None:
    """Test loading non-existent configuration file"""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")


def test_config_validation() -> None:
    """Test configuration validation"""
    config = load_config("development")

    # Validate cache config
    assert config.inference_service.cache.enabled is True
    assert config.inference_service.cache.ttl_seconds > 0

    # Validate circuit breaker config
    assert 0 < config.inference_service.model_server.circuit_breaker.failure_threshold <= 1
    assert config.inference_service.model_server.circuit_breaker.recovery_timeout_seconds > 0

    # Validate drift thresholds
    assert config.drift_detector.thresholds.feature_drift_warning > 0
    assert (
        config.drift_detector.thresholds.feature_drift_critical
        > config.drift_detector.thresholds.feature_drift_warning
    )
