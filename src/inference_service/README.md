# Inference Service

The Inference Service is the main user-facing API for the Real-Time ML Inference Platform. It orchestrates the complete prediction pipeline including feature retrieval, model invocation, response formatting, and event logging.

## Features

- **FastAPI Framework**: High-performance async API with automatic OpenAPI documentation
- **Request Validation**: Pydantic models for robust input validation
- **Health Checks**: Kubernetes-ready liveness and readiness probes
- **Observability**: Prometheus metrics, structured logging, and request tracing
- **Error Handling**: Standardized error responses with correlation IDs
- **OpenAPI Specification**: Interactive API documentation at `/docs`

## API Endpoints

### Inference

#### `POST /v1/predict`

Submit features for model inference and receive predictions.

**Request Body:**
```json
{
  "model_id": "fraud_detection_v2",
  "model_version": "1.2.3",  // Optional, defaults to latest production
  "features": {
    "age": 30,
    "income": 50000,
    "credit_score": 720
  },
  "options": {  // Optional
    "timeout_ms": 5000,
    "include_explanation": false,
    "enable_cache": true
  }
}
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "fraud_detection_v2",
  "model_version": "1.2.3",
  "prediction": {
    "fraud_probability": 0.15,
    "risk_score": 25
  },
  "confidence": 0.92,
  "latency_ms": 45.2,
  "timestamp": "2024-01-15T10:30:00Z",
  "cached": true
}
```

**Error Response:**
```json
{
  "error_code": "MODEL_NOT_FOUND",
  "message": "Model 'fraud_detection_v2' not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "model_id": "fraud_detection_v2"
  }
}
```

### Health Checks

#### `GET /health`

Liveness probe - returns 200 if service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "inference-service"
}
```

#### `GET /ready`

Readiness probe - returns 200 if service is ready to accept traffic.

**Response:**
```json
{
  "status": "ready",
  "service": "inference-service",
  "checks": {
    "cache": "healthy",
    "model_server": "healthy",
    "event_stream": "healthy"
  }
}
```

### Documentation

#### `GET /docs`

Interactive Swagger UI documentation.

#### `GET /redoc`

Alternative ReDoc documentation.

#### `GET /openapi.json`

OpenAPI 3.0 specification in JSON format.

### Metrics

#### `GET /metrics`

Prometheus metrics endpoint.

**Metrics Exposed:**
- `inference_requests_total`: Counter of total inference requests (labels: model_id, status, cache_hit)
- `inference_latency_seconds`: Histogram of request latency (labels: model_id)

## Running the Service

### Development

```bash
# Using Python directly
python -m src.inference_service.app

# Using uvicorn with auto-reload
uvicorn src.inference_service.app:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Using uvicorn with multiple workers
uvicorn src.inference_service.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Docker

```bash
# Build image
docker build -t inference-service:latest -f docker/inference-service.Dockerfile .

# Run container
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e REDIS_HOST=redis \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  inference-service:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up inference-service

# View logs
docker-compose logs -f inference-service
```

## Configuration

The service is configured via YAML files in the `config/` directory and environment variables.

### Environment Variables

- `ENVIRONMENT`: Environment name (development, staging, production)
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `REDIS_PASSWORD`: Redis password
- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated Kafka brokers
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI

### Configuration File

See `config/development.yaml` for example configuration:

```yaml
environment: development

inference_service:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout_seconds: 5
  log_level: INFO
  
  cache:
    enabled: true
    ttl_seconds: 300
    max_retries: 2
  
  model_server:
    url: http://model-server:8001
    timeout_ms: 4000
    circuit_breaker:
      failure_threshold: 0.5
      recovery_timeout_seconds: 60
  
  event_stream:
    enabled: true
    buffer_size: 1000
    flush_interval_ms: 100
```

## Testing

```bash
# Run unit tests
pytest tests/unit/test_inference_service_app.py -v

# Run with coverage
pytest tests/unit/test_inference_service_app.py --cov=src.inference_service --cov-report=html

# Run integration tests
pytest tests/integration/test_inference_service_integration.py -v
```

## Architecture

The Inference Service orchestrates the following pipeline:

1. **Request Validation**: Validate incoming request using Pydantic models
2. **Feature Retrieval**: Fetch features from cache or primary data source
3. **Model Invocation**: Call Model Server for inference
4. **Response Formatting**: Format prediction results with metadata
5. **Event Logging**: Asynchronously publish inference event to Kafka

```
Client Request
     ↓
Request Validation
     ↓
Feature Retrieval (Cache → DB fallback)
     ↓
Model Server Call (with Circuit Breaker)
     ↓
Response Formatting
     ↓
Event Publishing (async)
     ↓
Client Response
```

## Error Codes

| Code | HTTP Status | Description | Retriable |
|------|-------------|-------------|-----------|
| `INVALID_REQUEST` | 400 | Request validation failed | No |
| `MODEL_NOT_FOUND` | 404 | Requested model not found | No |
| `FEATURE_VALIDATION_FAILED` | 400 | Feature validation failed | No |
| `INFERENCE_TIMEOUT` | 504 | Inference request timed out | Yes |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded | Yes |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable | Yes |
| `INTERNAL_ERROR` | 500 | Internal server error | No |

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

Key metrics:
- Request rate by model
- Latency percentiles (p50, p95, p99)
- Error rate by error code
- Cache hit ratio

### Logging

Structured JSON logs with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Prediction completed successfully",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "fraud_detection_v2",
  "latency_ms": 45.2
}
```

## Performance

- **Target Latency**: <100ms p95
- **Throughput**: 1000+ requests/second per instance
- **Concurrency**: Async workers for high concurrency
- **Caching**: Redis cache for sub-5ms feature retrieval

## Security

- **Authentication**: API key or JWT token validation (via API Gateway)
- **Rate Limiting**: 100 requests/minute per API key
- **TLS**: All traffic encrypted with TLS 1.3
- **Input Validation**: Strict Pydantic validation on all inputs
- **PII Redaction**: Automatic PII redaction in logs

## Next Steps

1. Implement feature retrieval with cache integration (Task 7.2)
2. Add Model Server client for inference (Task 7.2)
3. Implement circuit breaker pattern (Task 7.3)
4. Add Kafka event publisher (Task 8.3)
5. Implement authentication middleware (Task 11.1)

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Design Document](../../.kiro/specs/real-time-ml-inference-platform/design.md)
