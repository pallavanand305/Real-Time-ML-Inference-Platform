"""Configuration for Model Server"""

import os
from pydantic import BaseModel, Field


class ModelServerSettings(BaseModel):
    """Model Server configuration settings"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model settings
    max_models_in_memory: int = Field(default=10, description="Maximum models to keep in memory")
    model_cache_memory_threshold: float = Field(default=0.8, description="Memory threshold for LRU eviction")
    model_load_timeout_seconds: int = Field(default=30, description="Timeout for model loading")
    
    # Inference settings
    max_batch_size: int = Field(default=100, description="Maximum batch size for inference")
    inference_timeout_seconds: int = Field(default=5, description="Timeout for inference execution")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    
    # Model Registry settings
    mlflow_tracking_uri: str = Field(
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        description="MLflow tracking server URI"
    )
    s3_bucket: str = Field(
        default=os.getenv("AWS_S3_BUCKET", "ml-models"),
        description="S3 bucket for model artifacts"
    )
    s3_region: str = Field(
        default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        description="AWS region"
    )
    
    # Health check settings
    health_check_interval_seconds: int = Field(default=30, description="Health check interval")
    
    class Config:
        env_prefix = "MODEL_SERVER_"


def get_settings() -> ModelServerSettings:
    """Get Model Server settings from environment variables"""
    return ModelServerSettings()
