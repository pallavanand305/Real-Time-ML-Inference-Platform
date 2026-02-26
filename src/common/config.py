"""Configuration management for the ML inference platform"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Cache configuration"""

    enabled: bool = True
    ttl_seconds: int = 300
    max_retries: int = 2


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""

    failure_threshold: float = 0.5
    recovery_timeout_seconds: int = 60


class ModelServerConfig(BaseModel):
    """Model server configuration"""

    url: str
    timeout_ms: int = 4000
    circuit_breaker: CircuitBreakerConfig


class EventStreamConfig(BaseModel):
    """Event stream configuration"""

    enabled: bool = True
    buffer_size: int = 1000
    flush_interval_ms: int = 100


class InferenceServiceConfig(BaseModel):
    """Inference service configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout_seconds: int = 5
    log_level: str = "INFO"
    cache: CacheConfig
    model_server: ModelServerConfig
    event_stream: EventStreamConfig


class RedisConfig(BaseModel):
    """Redis configuration"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class KafkaTopicsConfig(BaseModel):
    """Kafka topics configuration"""

    inference_events: str
    drift_alerts: str


class KafkaProducerConfig(BaseModel):
    """Kafka producer configuration"""

    acks: int = 1
    compression_type: str = "snappy"
    max_in_flight_requests: int = 5
    retries: int = 3
    linger_ms: int = 10
    batch_size: int = 16384


class KafkaConfig(BaseModel):
    """Kafka configuration"""

    bootstrap_servers: list[str]
    topics: KafkaTopicsConfig
    producer: KafkaProducerConfig


class PostgresConfig(BaseModel):
    """PostgreSQL configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "mlflow"
    user: str = "mlflow"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


class MLflowConfig(BaseModel):
    """MLflow configuration"""

    tracking_uri: str
    artifact_root: str


class S3Config(BaseModel):
    """S3 configuration"""

    bucket: str
    region: str = "us-east-1"


class DriftThresholdsConfig(BaseModel):
    """Drift detection thresholds"""

    feature_drift_warning: float = 0.1
    feature_drift_critical: float = 0.3
    prediction_drift_warning: float = 0.1
    prediction_drift_critical: float = 0.25


class DriftDetectorConfig(BaseModel):
    """Drift detector configuration"""

    check_interval_seconds: int = 3600
    baseline_window_hours: int = 168
    current_window_hours: int = 24
    thresholds: DriftThresholdsConfig


class PrometheusConfig(BaseModel):
    """Prometheus configuration"""

    port: int = 9090


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""

    prometheus: PrometheusConfig
    metrics_interval_seconds: int = 15


class Config(BaseModel):
    """Main configuration"""

    model_config = {"protected_namespaces": ()}

    environment: str
    inference_service: InferenceServiceConfig
    redis: RedisConfig
    kafka: KafkaConfig
    postgres: PostgresConfig
    mlflow: MLflowConfig
    s3: S3Config
    drift_detector: DriftDetectorConfig
    monitoring: MonitoringConfig


def load_config(environment: str | None = None) -> Config:
    """
    Load configuration from YAML file based on environment.

    Args:
        environment: Environment name (development, staging, production).
                    If None, reads from ENVIRONMENT env var or defaults to development.

    Returns:
        Config object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    config_dir = Path(__file__).parent.parent.parent / "config"
    config_file = config_dir / f"{environment}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Override with environment variables if present
    config_data = _override_with_env_vars(config_data)

    return Config(**config_data)


def _override_with_env_vars(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables"""
    # Redis overrides
    if "REDIS_HOST" in os.environ:
        config_data.setdefault("redis", {})["host"] = os.environ["REDIS_HOST"]
    if "REDIS_PORT" in os.environ:
        config_data.setdefault("redis", {})["port"] = int(os.environ["REDIS_PORT"])
    if "REDIS_PASSWORD" in os.environ:
        config_data.setdefault("redis", {})["password"] = os.environ["REDIS_PASSWORD"]

    # Kafka overrides
    if "KAFKA_BOOTSTRAP_SERVERS" in os.environ:
        servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"].split(",")
        config_data.setdefault("kafka", {})["bootstrap_servers"] = servers

    # PostgreSQL overrides
    if "POSTGRES_HOST" in os.environ:
        config_data.setdefault("postgres", {})["host"] = os.environ["POSTGRES_HOST"]
    if "POSTGRES_PASSWORD" in os.environ:
        config_data.setdefault("postgres", {})["password"] = os.environ["POSTGRES_PASSWORD"]

    # MLflow overrides
    if "MLFLOW_TRACKING_URI" in os.environ:
        config_data.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

    # S3 overrides
    if "AWS_S3_BUCKET" in os.environ:
        config_data.setdefault("s3", {})["bucket"] = os.environ["AWS_S3_BUCKET"]
    if "AWS_DEFAULT_REGION" in os.environ:
        config_data.setdefault("s3", {})["region"] = os.environ["AWS_DEFAULT_REGION"]

    return config_data
