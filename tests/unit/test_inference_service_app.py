"""Unit tests for Inference Service FastAPI application"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.inference_service.app import app
from src.common.models import PredictRequest, PredictOptions


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_config():
    """Mock configuration"""
    with patch("src.inference_service.app.load_config") as mock:
        config = MagicMock()
        config.environment = "test"
        config.inference_service.host = "0.0.0.0"
        config.inference_service.port = 8000
        config.inference_service.workers = 1
        config.inference_service.log_level = "INFO"
        mock.return_value = config
        yield config


def test_health_check(client):
    """Test health check endpoint returns 200"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "inference-service"


def test_readiness_check(client):
    """Test readiness check endpoint returns status"""
    response = client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "inference-service"
    assert "checks" in data


def test_predict_endpoint_valid_request(client):
    """Test predict endpoint with valid request"""
    request_data = {
        "model_id": "test_model",
        "features": {
            "age": 30,
            "income": 50000,
            "credit_score": 720
        }
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "request_id" in data
    assert data["model_id"] == "test_model"
    assert "model_version" in data
    assert "prediction" in data
    assert "latency_ms" in data
    assert "timestamp" in data
    assert isinstance(data["cached"], bool)
    
    # Verify request ID header
    assert "X-Request-ID" in response.headers


def test_predict_endpoint_with_model_version(client):
    """Test predict endpoint with specific model version"""
    request_data = {
        "model_id": "test_model",
        "model_version": "v1.2.3",
        "features": {
            "feature1": 1.0,
            "feature2": 2.0
        }
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.2.3"


def test_predict_endpoint_with_options(client):
    """Test predict endpoint with prediction options"""
    request_data = {
        "model_id": "test_model",
        "features": {"x": 1, "y": 2},
        "options": {
            "timeout_ms": 3000,
            "include_explanation": True,
            "enable_cache": False
        }
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 200


def test_predict_endpoint_empty_features(client):
    """Test predict endpoint rejects empty features"""
    request_data = {
        "model_id": "test_model",
        "features": {}
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    # Should return validation error
    assert response.status_code == 422


def test_predict_endpoint_missing_model_id(client):
    """Test predict endpoint rejects missing model_id"""
    request_data = {
        "features": {"x": 1}
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 422


def test_predict_endpoint_invalid_model_id(client):
    """Test predict endpoint rejects invalid model_id"""
    request_data = {
        "model_id": "",  # Empty model_id
        "features": {"x": 1}
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 422


def test_predict_endpoint_invalid_model_id_characters(client):
    """Test predict endpoint rejects model_id with invalid characters"""
    request_data = {
        "model_id": "model@#$%",  # Invalid characters
        "features": {"x": 1}
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 422


def test_predict_endpoint_missing_features(client):
    """Test predict endpoint rejects missing features"""
    request_data = {
        "model_id": "test_model"
    }
    
    response = client.post("/v1/predict", json=request_data)
    
    assert response.status_code == 422


def test_predict_endpoint_invalid_timeout(client):
    """Test predict endpoint rejects invalid timeout values"""
    # Timeout too low
    request_data = {
        "model_id": "test_model",
        "features": {"x": 1},
        "options": {"timeout_ms": 50}  # Below minimum of 100
    }
    
    response = client.post("/v1/predict", json=request_data)
    assert response.status_code == 422
    
    # Timeout too high
    request_data["options"]["timeout_ms"] = 40000  # Above maximum of 30000
    response = client.post("/v1/predict", json=request_data)
    assert response.status_code == 422


def test_predict_endpoint_invalid_confidence(client):
    """Test that confidence is validated in response model"""
    # This tests the response model validation
    from src.common.models import PredictResponse
    from datetime import datetime
    
    # Valid confidence
    response = PredictResponse(
        request_id="test",
        model_id="test_model",
        model_version="v1",
        prediction={"result": 1},
        confidence=0.85,
        latency_ms=100.0,
        timestamp=datetime.utcnow()
    )
    assert response.confidence == 0.85
    
    # Invalid confidence (too high)
    with pytest.raises(Exception):
        PredictResponse(
            request_id="test",
            model_id="test_model",
            model_version="v1",
            prediction={"result": 1},
            confidence=1.5,  # Above 1.0
            latency_ms=100.0,
            timestamp=datetime.utcnow()
        )
    
    # Invalid confidence (negative)
    with pytest.raises(Exception):
        PredictResponse(
            request_id="test",
            model_id="test_model",
            model_version="v1",
            prediction={"result": 1},
            confidence=-0.1,  # Below 0.0
            latency_ms=100.0,
            timestamp=datetime.utcnow()
        )


def test_openapi_spec_available(client):
    """Test that OpenAPI specification is available"""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    spec = response.json()
    
    # Verify basic OpenAPI structure
    assert "openapi" in spec
    assert "info" in spec
    assert spec["info"]["title"] == "ML Inference Service"
    assert "paths" in spec
    assert "/v1/predict" in spec["paths"]
    assert "/health" in spec["paths"]
    assert "/ready" in spec["paths"]


def test_swagger_ui_available(client):
    """Test that Swagger UI is available at /docs"""
    response = client.get("/docs")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_available(client):
    """Test that ReDoc is available at /redoc"""
    response = client.get("/redoc")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_metrics_endpoint_available(client):
    """Test that Prometheus metrics endpoint is available"""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    content = response.text
    
    # Verify Prometheus metrics format
    assert "inference_requests_total" in content
    assert "inference_latency_seconds" in content


def test_request_id_middleware(client):
    """Test that request ID is added to all responses"""
    response = client.get("/health")
    
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    
    # Verify it's a valid UUID format
    import uuid
    try:
        uuid.UUID(request_id)
    except ValueError:
        pytest.fail("Request ID is not a valid UUID")


def test_cors_headers(client):
    """Test CORS headers if configured"""
    # This is a placeholder - CORS would need to be configured in the app
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_request_model_validation():
    """Test PredictRequest model validation"""
    # Valid request
    request = PredictRequest(
        model_id="test_model",
        features={"x": 1, "y": 2}
    )
    assert request.model_id == "test_model"
    assert request.features == {"x": 1, "y": 2}
    
    # Empty features should fail
    with pytest.raises(Exception):
        PredictRequest(
            model_id="test_model",
            features={}
        )
    
    # Empty model_id should fail
    with pytest.raises(Exception):
        PredictRequest(
            model_id="",
            features={"x": 1}
        )
    
    # Invalid model_id characters should fail
    with pytest.raises(Exception):
        PredictRequest(
            model_id="model@#$",
            features={"x": 1}
        )


def test_predict_options_validation():
    """Test PredictOptions model validation"""
    # Valid options
    options = PredictOptions(
        timeout_ms=5000,
        include_explanation=True,
        enable_cache=False
    )
    assert options.timeout_ms == 5000
    assert options.include_explanation is True
    assert options.enable_cache is False
    
    # Default values
    options = PredictOptions()
    assert options.timeout_ms == 5000
    assert options.include_explanation is False
    assert options.enable_cache is True
    
    # Invalid timeout (too low)
    with pytest.raises(Exception):
        PredictOptions(timeout_ms=50)
    
    # Invalid timeout (too high)
    with pytest.raises(Exception):
        PredictOptions(timeout_ms=40000)


def test_error_response_format(client):
    """Test that error responses follow the standard format"""
    # Trigger a validation error
    response = client.post("/v1/predict", json={"model_id": "test"})
    
    assert response.status_code == 422
    
    # FastAPI validation errors have a different format
    # but our custom exception handlers should format other errors correctly


def test_concurrent_requests(client):
    """Test handling multiple concurrent requests"""
    import concurrent.futures
    
    def make_request():
        request_data = {
            "model_id": "test_model",
            "features": {"x": 1, "y": 2}
        }
        return client.post("/v1/predict", json=request_data)
    
    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    
    # All should have unique request IDs
    request_ids = [r.headers["X-Request-ID"] for r in responses]
    assert len(request_ids) == len(set(request_ids))
