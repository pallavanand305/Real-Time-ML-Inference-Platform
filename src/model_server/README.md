# Model Server

The Model Server is a FastAPI-based service responsible for loading ML models and executing inference requests.

## Features

- **Model Management**: Load, list, and unload models dynamically
- **Inference Execution**: Execute predictions with batch support
- **Health Checks**: Liveness and readiness probes for Kubernetes
- **Error Handling**: Standardized error responses with request tracing
- **Middleware**: Request ID tracking, CORS support, and request logging

## API Endpoints

### Health Checks

- `GET /health` - Liveness probe (returns 200 if service is running)
- `GET /ready` - Readiness probe (returns 200 if models are loaded)

### Model Management

- `POST /v1/models/load` - Load a model from the registry
- `GET /v1/models` - List all loaded models
- `DELETE /v1/models/{model_id}/{version}` - Unload a specific model

### Inference

- `POST /v1/infer` - Execute model inference

## Running the Server

### Development

```bash
# Run with default settings
python -m src.model_server

# Or use uvicorn directly
uvicorn src.model_server.app:app --reload --port 8001
```

### Production

```bash
# Run with multiple workers
uvicorn src.model_server.app:app --host 0.0.0.0 --port 8001 --workers 4
```

### Environment Variables

- `MODEL_SERVER_HOST` - Server host (default: 0.0.0.0)
- `MODEL_SERVER_PORT` - Server port (default: 8001)
- `MODEL_SERVER_WORKERS` - Number of workers (default: 4)
- `MODEL_SERVER_LOG_LEVEL` - Logging level (default: INFO)
- `MLFLOW_TRACKING_URI` - MLflow tracking server URI
- `AWS_S3_BUCKET` - S3 bucket for model artifacts

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
- OpenAPI JSON: http://localhost:8001/openapi.json

## Example Usage

### Load a Model

```bash
curl -X POST http://localhost:8001/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud_detection",
    "version": "2.0.0",
    "registry_uri": "s3://bucket/models/fraud_detection/2.0.0"
  }'
```

### Execute Inference

```bash
curl -X POST http://localhost:8001/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud_detection",
    "model_version": "2.0.0",
    "inputs": [
      {"feature1": 1.0, "feature2": 2.0},
      {"feature1": 3.0, "feature2": 4.0}
    ]
  }'
```

### List Loaded Models

```bash
curl http://localhost:8001/v1/models
```

## Architecture

The Model Server follows a simple architecture:

1. **FastAPI Application**: Handles HTTP requests and routing
2. **Model Cache**: In-memory storage for loaded models (LRU eviction)
3. **Model Loader**: Downloads and loads models from MLflow/S3
4. **Inference Engine**: Executes predictions using loaded models

## Next Steps

The current implementation provides the API structure with mock responses. Future tasks will add:

- Actual model loading from MLflow registry
- ONNX Runtime integration for inference
- Model caching with LRU eviction
- GPU acceleration support
- Schema validation for inputs
- Batch inference optimization
