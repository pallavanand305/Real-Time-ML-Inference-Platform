"""Tests for Pydantic models"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.common.models import (
    ErrorCode,
    ErrorResponse,
    PredictOptions,
    PredictRequest,
    PredictResponse,
)


class TestPredictOptions:
    """Tests for PredictOptions model"""

    def test_default_values(self) -> None:
        """Test default values for PredictOptions"""
        options = PredictOptions()

        assert options.timeout_ms == 5000
        assert options.include_explanation is False
        assert options.enable_cache is True

    def test_custom_values(self) -> None:
        """Test custom values for PredictOptions"""
        options = PredictOptions(
            timeout_ms=3000,
            include_explanation=True,
            enable_cache=False
        )

        assert options.timeout_ms == 3000
        assert options.include_explanation is True
        assert options.enable_cache is False

    def test_timeout_validation_min(self) -> None:
        """Test timeout_ms minimum validation"""
        with pytest.raises(ValidationError) as exc_info:
            PredictOptions(timeout_ms=50)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout_ms",)
        assert "greater than or equal to 100" in errors[0]["msg"]

    def test_timeout_validation_max(self) -> None:
        """Test timeout_ms maximum validation"""
        with pytest.raises(ValidationError) as exc_info:
            PredictOptions(timeout_ms=40000)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout_ms",)
        assert "less than or equal to 30000" in errors[0]["msg"]

    def test_timeout_boundary_values(self) -> None:
        """Test timeout_ms boundary values"""
        # Minimum boundary
        options_min = PredictOptions(timeout_ms=100)
        assert options_min.timeout_ms == 100

        # Maximum boundary
        options_max = PredictOptions(timeout_ms=30000)
        assert options_max.timeout_ms == 30000


class TestPredictRequest:
    """Tests for PredictRequest model"""

    def test_valid_request(self) -> None:
        """Test valid PredictRequest"""
        request = PredictRequest(
            model_id="fraud_detection_v2",
            features={"age": 32, "amount": 150.50}
        )

        assert request.model_id == "fraud_detection_v2"
        assert request.features == {"age": 32, "amount": 150.50}
        assert request.model_version is None
        assert request.options is None

    def test_request_with_all_fields(self) -> None:
        """Test PredictRequest with all optional fields"""
        options = PredictOptions(timeout_ms=3000)
        request = PredictRequest(
            model_id="fraud_detection_v2",
            model_version="1.2.0",
            features={"age": 32, "amount": 150.50},
            options=options
        )

        assert request.model_id == "fraud_detection_v2"
        assert request.model_version == "1.2.0"
        assert request.features == {"age": 32, "amount": 150.50}
        assert request.options.timeout_ms == 3000

    def test_empty_features_validation(self) -> None:
        """Test validation fails for empty features"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(
                model_id="fraud_detection_v2",
                features={}
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("features",)
        assert "Features cannot be empty" in str(errors[0]["msg"])

    def test_missing_model_id(self) -> None:
        """Test validation fails for missing model_id"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(features={"age": 32})

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("model_id",) for error in errors)

    def test_missing_features(self) -> None:
        """Test validation fails for missing features"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(model_id="fraud_detection_v2")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("features",) for error in errors)

    def test_model_id_whitespace_trimming(self) -> None:
        """Test model_id whitespace is trimmed"""
        request = PredictRequest(
            model_id="  fraud_detection_v2  ",
            features={"age": 32}
        )

        assert request.model_id == "fraud_detection_v2"

    def test_model_id_empty_string(self) -> None:
        """Test validation fails for empty model_id"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(
                model_id="",
                features={"age": 32}
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("model_id",)
        assert "Model ID cannot be empty" in str(errors[0]["msg"])

    def test_model_id_whitespace_only(self) -> None:
        """Test validation fails for whitespace-only model_id"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(
                model_id="   ",
                features={"age": 32}
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("model_id",)
        assert "Model ID cannot be empty" in str(errors[0]["msg"])

    def test_model_id_invalid_characters(self) -> None:
        """Test validation fails for invalid characters in model_id"""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(
                model_id="fraud@detection#v2",
                features={"age": 32}
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("model_id",)
        assert "alphanumeric characters" in str(errors[0]["msg"])

    def test_model_id_valid_characters(self) -> None:
        """Test model_id accepts valid characters"""
        valid_ids = [
            "fraud_detection_v2",
            "fraud-detection-v2",
            "FraudDetectionV2",
            "fraud123",
            "model_v1-2-3"
        ]

        for model_id in valid_ids:
            request = PredictRequest(
                model_id=model_id,
                features={"age": 32}
            )
            assert request.model_id == model_id

    def test_features_with_various_types(self) -> None:
        """Test features can contain various data types"""
        request = PredictRequest(
            model_id="test_model",
            features={
                "int_feature": 42,
                "float_feature": 3.14,
                "string_feature": "test",
                "bool_feature": True,
                "list_feature": [1, 2, 3],
                "dict_feature": {"nested": "value"}
            }
        )

        assert request.features["int_feature"] == 42
        assert request.features["float_feature"] == 3.14
        assert request.features["string_feature"] == "test"
        assert request.features["bool_feature"] is True
        assert request.features["list_feature"] == [1, 2, 3]
        assert request.features["dict_feature"] == {"nested": "value"}


class TestPredictResponse:
    """Tests for PredictResponse model"""

    def test_valid_response(self) -> None:
        """Test valid PredictResponse"""
        response = PredictResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            model_id="fraud_detection_v2",
            model_version="1.2.0",
            prediction={"fraud": False, "score": 0.15},
            latency_ms=45.5
        )

        assert response.request_id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.model_id == "fraud_detection_v2"
        assert response.model_version == "1.2.0"
        assert response.prediction == {"fraud": False, "score": 0.15}
        assert response.latency_ms == 45.5
        assert response.confidence is None
        assert response.cached is False
        assert isinstance(response.timestamp, datetime)

    def test_response_with_confidence(self) -> None:
        """Test PredictResponse with confidence score"""
        response = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=1,
            confidence=0.95,
            latency_ms=50.0
        )

        assert response.confidence == 0.95

    def test_response_with_cached_flag(self) -> None:
        """Test PredictResponse with cached flag"""
        response = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=1,
            latency_ms=50.0,
            cached=True
        )

        assert response.cached is True

    def test_confidence_validation_min(self) -> None:
        """Test confidence minimum validation"""
        with pytest.raises(ValidationError) as exc_info:
            PredictResponse(
                request_id="test-id",
                model_id="test_model",
                model_version="1.0.0",
                prediction=1,
                confidence=-0.1,
                latency_ms=50.0
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("confidence",)
        assert "greater than or equal to 0" in errors[0]["msg"]

    def test_confidence_validation_max(self) -> None:
        """Test confidence maximum validation"""
        with pytest.raises(ValidationError) as exc_info:
            PredictResponse(
                request_id="test-id",
                model_id="test_model",
                model_version="1.0.0",
                prediction=1,
                confidence=1.5,
                latency_ms=50.0
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("confidence",)
        assert "less than or equal to 1" in errors[0]["msg"]

    def test_confidence_boundary_values(self) -> None:
        """Test confidence boundary values"""
        # Minimum boundary
        response_min = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=1,
            confidence=0.0,
            latency_ms=50.0
        )
        assert response_min.confidence == 0.0

        # Maximum boundary
        response_max = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=1,
            confidence=1.0,
            latency_ms=50.0
        )
        assert response_max.confidence == 1.0

    def test_latency_validation_negative(self) -> None:
        """Test latency_ms cannot be negative"""
        with pytest.raises(ValidationError) as exc_info:
            PredictResponse(
                request_id="test-id",
                model_id="test_model",
                model_version="1.0.0",
                prediction=1,
                latency_ms=-10.0
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("latency_ms",)
        assert "greater than or equal to 0" in errors[0]["msg"]

    def test_prediction_various_types(self) -> None:
        """Test prediction can be various types"""
        # Scalar prediction
        response1 = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=42,
            latency_ms=50.0
        )
        assert response1.prediction == 42

        # Dict prediction
        response2 = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction={"class": "A", "score": 0.9},
            latency_ms=50.0
        )
        assert response2.prediction == {"class": "A", "score": 0.9}

        # List prediction
        response3 = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction=[0.1, 0.2, 0.7],
            latency_ms=50.0
        )
        assert response3.prediction == [0.1, 0.2, 0.7]


class TestErrorResponse:
    """Tests for ErrorResponse model"""

    def test_valid_error_response(self) -> None:
        """Test valid ErrorResponse"""
        error = ErrorResponse(
            error_code=ErrorCode.INVALID_REQUEST,
            message="Request validation failed",
            request_id="test-request-id"
        )

        assert error.error_code == ErrorCode.INVALID_REQUEST
        assert error.message == "Request validation failed"
        assert error.request_id == "test-request-id"
        assert error.details is None
        assert isinstance(error.timestamp, datetime)

    def test_error_response_with_details(self) -> None:
        """Test ErrorResponse with details"""
        error = ErrorResponse(
            error_code=ErrorCode.FEATURE_VALIDATION_FAILED,
            message="Feature validation failed",
            request_id="test-request-id",
            details={
                "missing_features": ["age", "income"],
                "invalid_features": {"amount": "must be positive"}
            }
        )

        assert error.details is not None
        assert error.details["missing_features"] == ["age", "income"]
        assert error.details["invalid_features"]["amount"] == "must be positive"

    def test_all_error_codes(self) -> None:
        """Test all error codes are valid"""
        error_codes = [
            ErrorCode.INVALID_REQUEST,
            ErrorCode.MODEL_NOT_FOUND,
            ErrorCode.FEATURE_VALIDATION_FAILED,
            ErrorCode.INFERENCE_TIMEOUT,
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.RATE_LIMIT_EXCEEDED,
            ErrorCode.UNAUTHORIZED,
        ]

        for code in error_codes:
            error = ErrorResponse(
                error_code=code,
                message=f"Test error for {code}",
                request_id="test-id"
            )
            assert error.error_code == code

    def test_error_code_enum_values(self) -> None:
        """Test ErrorCode enum has expected values"""
        assert ErrorCode.INVALID_REQUEST.value == "INVALID_REQUEST"
        assert ErrorCode.MODEL_NOT_FOUND.value == "MODEL_NOT_FOUND"
        assert ErrorCode.FEATURE_VALIDATION_FAILED.value == "FEATURE_VALIDATION_FAILED"
        assert ErrorCode.INFERENCE_TIMEOUT.value == "INFERENCE_TIMEOUT"
        assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"
        assert ErrorCode.SERVICE_UNAVAILABLE.value == "SERVICE_UNAVAILABLE"
        assert ErrorCode.RATE_LIMIT_EXCEEDED.value == "RATE_LIMIT_EXCEEDED"
        assert ErrorCode.UNAUTHORIZED.value == "UNAUTHORIZED"

    def test_missing_required_fields(self) -> None:
        """Test validation fails for missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="Test error"
                # missing request_id
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("request_id",) for error in errors)


class TestModelSerialization:
    """Tests for model serialization and deserialization"""

    def test_predict_request_json_serialization(self) -> None:
        """Test PredictRequest JSON serialization"""
        request = PredictRequest(
            model_id="test_model",
            model_version="1.0.0",
            features={"age": 32, "amount": 150.50}
        )

        json_data = request.model_dump()
        assert json_data["model_id"] == "test_model"
        assert json_data["model_version"] == "1.0.0"
        assert json_data["features"] == {"age": 32, "amount": 150.50}

    def test_predict_request_json_deserialization(self) -> None:
        """Test PredictRequest JSON deserialization"""
        json_data = {
            "model_id": "test_model",
            "features": {"age": 32}
        }

        request = PredictRequest(**json_data)
        assert request.model_id == "test_model"
        assert request.features == {"age": 32}

    def test_predict_response_json_serialization(self) -> None:
        """Test PredictResponse JSON serialization"""
        response = PredictResponse(
            request_id="test-id",
            model_id="test_model",
            model_version="1.0.0",
            prediction={"result": "positive"},
            confidence=0.95,
            latency_ms=45.5,
            cached=True
        )

        json_data = response.model_dump()
        assert json_data["request_id"] == "test-id"
        assert json_data["model_id"] == "test_model"
        assert json_data["prediction"] == {"result": "positive"}
        assert json_data["confidence"] == 0.95
        assert json_data["cached"] is True

    def test_error_response_json_serialization(self) -> None:
        """Test ErrorResponse JSON serialization"""
        error = ErrorResponse(
            error_code=ErrorCode.INVALID_REQUEST,
            message="Test error",
            request_id="test-id",
            details={"field": "value"}
        )

        json_data = error.model_dump()
        assert json_data["error_code"] == "INVALID_REQUEST"
        assert json_data["message"] == "Test error"
        assert json_data["request_id"] == "test-id"
        assert json_data["details"] == {"field": "value"}
