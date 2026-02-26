"""FastAPI application for Model Server"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.common.models import ErrorCode, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Request/Response Models for Model Server
class InferRequest(BaseModel):
    """Request model for inference"""
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    inputs: List[Dict[str, Any]] = Field(..., description="Batch of feature vectors")


class ModelMetadata(BaseModel):
    """Model metadata"""
    model_id: str
    version: str
    framework: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]


class InferResponse(BaseModel):
    """Response model for inference"""
    predictions: List[Any] = Field(..., description="Model predictions")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    inference_time_ms: float = Field(..., description="Inference execution time")


class LoadModelRequest(BaseModel):
    """Request model for loading a model"""
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    registry_uri: str = Field(..., description="Model registry URI")


class LoadModelResponse(BaseModel):
    """Response model for model loading"""
    model_id: str
    version: str
    status: str
    message: str


class ModelInfo(BaseModel):
    """Information about a loaded model"""
    model_id: str
    version: str
    framework: str
    loaded_at: str
    memory_mb: float


class ListModelsResponse(BaseModel):
    """Response model for listing models"""
    models: List[ModelInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    """Readiness check response"""
    ready: bool
    models_loaded: int
    message: str


# Global state for loaded models (will be replaced with actual model cache)
loaded_models: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Model Server starting up...")
    # TODO: Load production models on startup
    yield
    # Shutdown
    logger.info("Model Server shutting down...")
    # TODO: Cleanup resources


# Create FastAPI application
app = FastAPI(
    title="Model Server API",
    description="ML Model serving API for inference execution",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"request_id": request_id}
    )
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration_ms:.2f}ms",
        extra={"request_id": request_id, "duration_ms": duration_ms}
    )
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error response"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Map HTTP status to error code
    error_code_map = {
        400: ErrorCode.INVALID_REQUEST,
        404: ErrorCode.MODEL_NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        503: ErrorCode.SERVICE_UNAVAILABLE,
        504: ErrorCode.INFERENCE_TIMEOUT,
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    
    error_response = ErrorResponse(
        error_code=error_code,
        message=exc.detail,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )
    
    error_response = ErrorResponse(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An internal error occurred",
        request_id=request_id,
        details={"error": str(exc)}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode='json')
    )


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for liveness probe.
    Returns 200 if the service is running.
    """
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint for readiness probe.
    Returns 200 if the service is ready to accept requests.
    """
    # Check if at least one model is loaded
    models_count = len(loaded_models)
    is_ready = models_count > 0
    
    return ReadyResponse(
        ready=is_ready,
        models_loaded=models_count,
        message="Ready" if is_ready else "No models loaded"
    )


# Model inference endpoint
@app.post("/v1/infer", response_model=InferResponse, tags=["Inference"])
async def infer(request: InferRequest, req: Request):
    """
    Execute model inference on input features.
    
    - **model_id**: Model identifier
    - **model_version**: Specific model version
    - **inputs**: List of feature dictionaries for batch inference
    
    Returns predictions for all inputs.
    """
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        f"Inference request: model={request.model_id} version={request.model_version} "
        f"batch_size={len(request.inputs)}",
        extra={"request_id": request_id}
    )
    
    # Check if model is loaded
    model_key = f"{request.model_id}:{request.model_version}"
    if model_key not in loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} version {request.model_version} not found"
        )
    
    # TODO: Implement actual inference logic
    # For now, return mock predictions
    model_info = loaded_models[model_key]
    predictions = [{"prediction": 0.5, "class": "positive"} for _ in request.inputs]
    
    inference_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"Inference completed: duration={inference_time_ms:.2f}ms",
        extra={"request_id": request_id}
    )
    
    return InferResponse(
        predictions=predictions,
        model_metadata=ModelMetadata(
            model_id=request.model_id,
            version=request.model_version,
            framework=model_info.get("framework", "unknown"),
            input_schema=model_info.get("input_schema", {}),
            output_schema=model_info.get("output_schema", {})
        ),
        inference_time_ms=inference_time_ms
    )


# Model management endpoints
@app.post("/v1/models/load", response_model=LoadModelResponse, tags=["Model Management"])
async def load_model(request: LoadModelRequest, req: Request):
    """
    Load a model into memory from the model registry.
    
    - **model_id**: Model identifier
    - **version**: Model version to load
    - **registry_uri**: URI to the model registry
    
    Returns status of the load operation.
    """
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        f"Loading model: {request.model_id} version {request.version}",
        extra={"request_id": request_id}
    )
    
    model_key = f"{request.model_id}:{request.version}"
    
    # Check if model is already loaded
    if model_key in loaded_models:
        return LoadModelResponse(
            model_id=request.model_id,
            version=request.version,
            status="already_loaded",
            message=f"Model {request.model_id} version {request.version} is already loaded"
        )
    
    # TODO: Implement actual model loading from registry
    # For now, add mock model info
    from datetime import datetime
    
    loaded_models[model_key] = {
        "model_id": request.model_id,
        "version": request.version,
        "framework": "onnx",
        "loaded_at": datetime.utcnow().isoformat(),
        "input_schema": {"feature1": "float", "feature2": "float"},
        "output_schema": {"prediction": "float", "class": "string"},
        "memory_mb": 100.0
    }
    
    logger.info(
        f"Model loaded successfully: {request.model_id} version {request.version}",
        extra={"request_id": request_id}
    )
    
    return LoadModelResponse(
        model_id=request.model_id,
        version=request.version,
        status="loaded",
        message=f"Model {request.model_id} version {request.version} loaded successfully"
    )


@app.get("/v1/models", response_model=ListModelsResponse, tags=["Model Management"])
async def list_models():
    """
    List all currently loaded models.
    
    Returns information about all models in memory.
    """
    models = []
    for model_key, model_info in loaded_models.items():
        models.append(ModelInfo(
            model_id=model_info["model_id"],
            version=model_info["version"],
            framework=model_info["framework"],
            loaded_at=model_info["loaded_at"],
            memory_mb=model_info["memory_mb"]
        ))
    
    return ListModelsResponse(
        models=models,
        total=len(models)
    )


@app.delete("/v1/models/{model_id}/{version}", tags=["Model Management"])
async def unload_model(model_id: str, version: str, req: Request):
    """
    Unload a model from memory.
    
    - **model_id**: Model identifier
    - **version**: Model version to unload
    
    Returns 204 on success.
    """
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    
    model_key = f"{model_id}:{version}"
    
    if model_key not in loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} version {version} not found"
        )
    
    # TODO: Implement graceful model unloading (wait for in-flight requests)
    del loaded_models[model_key]
    
    logger.info(
        f"Model unloaded: {model_id} version {version}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_204_NO_CONTENT,
        content=None
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.model_server.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
