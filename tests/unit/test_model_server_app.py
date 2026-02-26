"""Unit tests for Model Server API"""

import pytest
from fastapi.testclient import TestClient

from src.model_server.app import app, loaded_models


@pytest.fixture
def client():
    """Create test client"""
    # Clear loaded models before each test
    loaded_models.clear()
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_readiness_check_no_models(self, client):
        """Test readiness check when no models are loaded"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["models_loaded"] == 0
        assert "No models loaded" in data["message"]
    
    def test_readiness_check_with_models(self, client):
        """Test readiness check when models are loaded"""
        # Load a model first
        load_response = client.post(
            "/v1/models/load",
            json={
                "model_id": "test_model",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/test_model/1.0.0"
            }
        )
        assert load_response.status_code == 200
        
        # Check readiness
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["models_loaded"] == 1


class TestModelManagement:
    """Tests for model management endpoints"""
    
    def test_load_model(self, client):
        """Test loading a model"""
        response = client.post(
            "/v1/models/load",
            json={
                "model_id": "fraud_detection",
                "version": "2.0.0",
                "registry_uri": "s3://bucket/models/fraud_detection/2.0.0"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "fraud_detection"
        assert data["version"] == "2.0.0"
        assert data["status"] == "loaded"
        assert "loaded successfully" in data["message"]
    
    def test_load_model_already_loaded(self, client):
        """Test loading a model that's already loaded"""
        # Load model first time
        client.post(
            "/v1/models/load",
            json={
                "model_id": "test_model",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/test_model/1.0.0"
            }
        )
        
        # Try to load again
        response = client.post(
            "/v1/models/load",
            json={
                "model_id": "test_model",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/test_model/1.0.0"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_loaded"
    
    def test_list_models_empty(self, client):
        """Test listing models when none are loaded"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["total"] == 0
    
    def test_list_models_with_loaded_models(self, client):
        """Test listing models after loading some"""
        # Load two models
        client.post(
            "/v1/models/load",
            json={
                "model_id": "model1",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/model1/1.0.0"
            }
        )
        client.post(
            "/v1/models/load",
            json={
                "model_id": "model2",
                "version": "2.0.0",
                "registry_uri": "s3://bucket/models/model2/2.0.0"
            }
        )
        
        # List models
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["models"]) == 2
        
        model_ids = [m["model_id"] for m in data["models"]]
        assert "model1" in model_ids
        assert "model2" in model_ids
    
    def test_unload_model(self, client):
        """Test unloading a model"""
        # Load model first
        client.post(
            "/v1/models/load",
            json={
                "model_id": "test_model",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/test_model/1.0.0"
            }
        )
        
        # Unload model
        response = client.delete("/v1/models/test_model/1.0.0")
        assert response.status_code == 204
        
        # Verify model is unloaded
        list_response = client.get("/v1/models")
        assert list_response.json()["total"] == 0
    
    def test_unload_model_not_found(self, client):
        """Test unloading a model that doesn't exist"""
        response = client.delete("/v1/models/nonexistent/1.0.0")
        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "MODEL_NOT_FOUND"


class TestInference:
    """Tests for inference endpoint"""
    
    def test_infer_model_not_loaded(self, client):
        """Test inference when model is not loaded"""
        response = client.post(
            "/v1/infer",
            json={
                "model_id": "nonexistent",
                "model_version": "1.0.0",
                "inputs": [{"feature1": 1.0, "feature2": 2.0}]
            }
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "MODEL_NOT_FOUND"
    
    def test_infer_success(self, client):
        """Test successful inference"""
        # Load model first
        client.post(
            "/v1/models/load",
            json={
                "model_id": "fraud_detection",
                "version": "2.0.0",
                "registry_uri": "s3://bucket/models/fraud_detection/2.0.0"
            }
        )
        
        # Make inference request
        response = client.post(
            "/v1/infer",
            json={
                "model_id": "fraud_detection",
                "model_version": "2.0.0",
                "inputs": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 3.0, "feature2": 4.0}
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert "model_metadata" in data
        assert data["model_metadata"]["model_id"] == "fraud_detection"
        assert data["model_metadata"]["version"] == "2.0.0"
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0
    
    def test_infer_batch(self, client):
        """Test batch inference with multiple inputs"""
        # Load model
        client.post(
            "/v1/models/load",
            json={
                "model_id": "test_model",
                "version": "1.0.0",
                "registry_uri": "s3://bucket/models/test_model/1.0.0"
            }
        )
        
        # Make batch inference request
        batch_size = 10
        inputs = [{"feature1": i, "feature2": i * 2} for i in range(batch_size)]
        
        response = client.post(
            "/v1/infer",
            json={
                "model_id": "test_model",
                "model_version": "1.0.0",
                "inputs": inputs
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == batch_size


class TestMiddleware:
    """Tests for middleware functionality"""
    
    def test_request_id_header(self, client):
        """Test that request ID is added to response headers"""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
    
    def test_custom_request_id(self, client):
        """Test that custom request ID is preserved"""
        custom_id = "test-request-123"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test invalid JSON in request body"""
        response = client.post(
            "/v1/models/load",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
