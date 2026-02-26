"""Pydantic models for API requests, responses, and domain objects"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Error Models
class ErrorCode(str, Enum):
    """Error codes for API responses"""
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    FEATURE_VALIDATION_FAILED = "FEATURE_VALIDATION_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error_code: ErrorCode = Field(..., description="Error code identifying the error type")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Request/Response Models
class PredictOptions(BaseModel):
    """Optional parameters for prediction requests"""
    timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=30000,
        description="Request timeout in milliseconds"
    )
    include_explanation: bool = Field(
        default=False,
        description="Whether to include model explanation in response"
    )
    enable_cache: bool = Field(
        default=True,
        description="Whether to use feature cache"
    )


class PredictRequest(BaseModel):
    """Request model for inference predictions"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str = Field(..., description="Model identifier")
    model_version: Optional[str] = Field(
        None,
        description="Specific model version, defaults to latest production version"
    )
    features: Dict[str, Any] = Field(..., description="Feature vector for prediction")
    options: Optional[PredictOptions] = Field(None, description="Optional prediction parameters")

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that features dictionary is not empty"""
        if not v:
            raise ValueError("Features cannot be empty")
        return v

    @field_validator('model_id')
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate model_id is not empty and contains valid characters"""
        # Trim whitespace first
        v = v.strip()
        
        if not v:
            raise ValueError("Model ID cannot be empty")
        if not all(c.isalnum() or c in ('_', '-') for c in v):
            raise ValueError("Model ID can only contain alphanumeric characters, underscores, and hyphens")
        return v


class PredictResponse(BaseModel):
    """Response model for inference predictions"""
    model_config = ConfigDict(protected_namespaces=())
    
    request_id: str = Field(..., description="Unique request identifier")
    model_id: str = Field(..., description="Model identifier used for prediction")
    model_version: str = Field(..., description="Model version used for prediction")
    prediction: Any = Field(..., description="Model prediction output")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score (0-1)"
    )
    latency_ms: float = Field(..., ge=0, description="Request processing latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    cached: bool = Field(default=False, description="Whether features were retrieved from cache")


# Feature Models
class FeatureSchema(BaseModel):
    """Schema definition for a single feature"""
    name: str = Field(..., description="Feature name")
    dtype: str = Field(..., description="Data type: int, float, string, bool, array")
    required: bool = Field(default=True, description="Whether feature is required")
    default: Optional[Any] = Field(None, description="Default value if not provided")
    description: Optional[str] = Field(None, description="Feature description")

    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate dtype is one of the allowed types"""
        valid_types = {"int", "float", "string", "bool", "array"}
        if v not in valid_types:
            raise ValueError(f"dtype must be one of {valid_types}")
        return v


class FeatureVector(BaseModel):
    """Feature vector with metadata"""
    entity_id: str = Field(..., description="Entity identifier (e.g., user_id, transaction_id)")
    entity_type: str = Field(..., description="Type of entity (e.g., user, transaction)")
    features: Dict[str, Any] = Field(..., description="Feature key-value pairs")
    computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when features were computed"
    )
    version: str = Field(..., description="Feature schema version")

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that features dictionary is not empty"""
        if not v:
            raise ValueError("Features cannot be empty")
        return v


# Model Models
class ResourceRequirements(BaseModel):
    """Resource requirements for model deployment"""
    cpu_cores: float = Field(default=1.0, ge=0.1, description="CPU cores required")
    memory_mb: int = Field(default=2048, ge=128, description="Memory in MB")
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs required")
    gpu_type: Optional[str] = Field(None, description="GPU type (e.g., nvidia-tesla-t4)")


class DeploymentStatus(str, Enum):
    """Model deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    RETIRED = "retired"


class ModelMetadata(BaseModel):
    """Metadata for a trained model"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version (semantic versioning)")
    framework: str = Field(..., description="ML framework (pytorch, tensorflow, sklearn, onnx)")
    input_schema: Dict[str, FeatureSchema] = Field(..., description="Expected input feature schema")
    output_schema: Dict[str, str] = Field(..., description="Output schema definition")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Model creation timestamp"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Training metrics (e.g., accuracy, f1_score)"
    )

    @field_validator('framework')
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate framework is one of the supported types"""
        valid_frameworks = {"pytorch", "tensorflow", "sklearn", "onnx", "xgboost", "lightgbm"}
        if v.lower() not in valid_frameworks:
            raise ValueError(f"framework must be one of {valid_frameworks}")
        return v.lower()


class ModelDeployment(BaseModel):
    """Model deployment configuration and status"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version being deployed")
    stage: str = Field(..., description="Deployment stage: development, staging, production")
    replicas: int = Field(default=1, ge=1, description="Number of replicas")
    resources: ResourceRequirements = Field(..., description="Resource requirements")
    deployed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Deployment timestamp"
    )
    deployed_by: str = Field(..., description="User who deployed the model")
    status: DeploymentStatus = Field(
        default=DeploymentStatus.PENDING,
        description="Current deployment status"
    )

    @field_validator('stage')
    @classmethod
    def validate_stage(cls, v: str) -> str:
        """Validate stage is one of the allowed values"""
        valid_stages = {"development", "staging", "production"}
        if v.lower() not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}")
        return v.lower()


# Event Models
class InferenceEvent(BaseModel):
    """Event logged for each inference request"""
    model_config = ConfigDict(protected_namespaces=())
    
    event_id: str = Field(..., description="Unique event identifier")
    request_id: str = Field(..., description="Request identifier for correlation")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version used")
    features: Dict[str, Any] = Field(..., description="Input features")
    prediction: Any = Field(..., description="Model prediction output")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    latency_ms: float = Field(..., ge=0, description="Inference latency in milliseconds")
    cache_hit: bool = Field(..., description="Whether features were retrieved from cache")
    client_id: str = Field(..., description="Client identifier")
    environment: str = Field(..., description="Environment: production, staging, development")

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values"""
        valid_envs = {"production", "staging", "development"}
        if v.lower() not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")
        return v.lower()


class DriftAlert(BaseModel):
    """Alert generated when model drift is detected"""
    model_config = ConfigDict(protected_namespaces=())
    
    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Alert timestamp"
    )
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    drift_type: str = Field(..., description="Type of drift: feature_drift, prediction_drift")
    metric: str = Field(..., description="Drift metric used: kl_divergence, psi")
    value: float = Field(..., description="Computed drift metric value")
    threshold: float = Field(..., description="Threshold that was exceeded")
    affected_features: list[str] = Field(..., description="List of features affected by drift")
    severity: str = Field(..., description="Alert severity: warning, critical")
    baseline_period: str = Field(..., description="Baseline time period (e.g., '7 days')")
    current_period: str = Field(..., description="Current time period (e.g., '24 hours')")

    @field_validator('drift_type')
    @classmethod
    def validate_drift_type(cls, v: str) -> str:
        """Validate drift_type is one of the allowed values"""
        valid_types = {"feature_drift", "prediction_drift"}
        if v.lower() not in valid_types:
            raise ValueError(f"drift_type must be one of {valid_types}")
        return v.lower()

    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is one of the supported types"""
        valid_metrics = {"kl_divergence", "psi", "ks_statistic", "wasserstein"}
        if v.lower() not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        return v.lower()

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values"""
        valid_severities = {"warning", "critical"}
        if v.lower() not in valid_severities:
            raise ValueError(f"severity must be one of {valid_severities}")
        return v.lower()
