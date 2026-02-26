# Real-Time ML Inference Platform

A production-grade real-time ML inference platform with FastAPI, Redis caching, Kafka event streaming, MLflow model registry, and comprehensive observability.

## Features

- **Low Latency**: Sub-100ms p95 inference latency through intelligent caching
- **High Throughput**: Support 1000+ requests per second per instance
- **Model Management**: Complete MLOps workflow with MLflow registry
- **Event Streaming**: Asynchronous inference logging with Kafka
- **Drift Detection**: Automated model degradation monitoring
- **Observability**: Comprehensive metrics with Prometheus and Grafana
- **Production Ready**: Security, compliance, CI/CD, and operational tooling

## Architecture

The platform consists of three main services:

1. **Inference Service**: FastAPI-based orchestration layer handling prediction requests
2. **Model Server**: ONNX Runtime-based model execution engine
3. **Drift Detector**: Statistical monitoring service for model degradation

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry (for dependency management)

## Quick Start

### 1. Install Dependencies

#### Option A: Using Poetry (Recommended)

```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies
poetry install
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Set Up Development Environment

#### With Poetry:
```bash
poetry run pre-commit install
```

#### With pip:
```bash
pre-commit install
```

### 3. Start Infrastructure Services

```bash
make docker-up
```

This starts:
- Redis (feature cache)
- PostgreSQL (MLflow backend)
- Kafka + Zookeeper (event streaming)
- MLflow (model registry)
- Prometheus (metrics)
- Grafana (visualization)

### 4. Run Tests

#### With Poetry:
```bash
poetry run pytest
```

#### With pip:
```bash
pytest
```

Or use the Makefile:
```bash
make test
```

## Development

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make type-check
```

### Project Structure

```
.
├── src/
│   ├── inference_service/    # FastAPI inference orchestration
│   ├── model_server/          # Model loading and execution
│   ├── drift_detector/        # Drift monitoring service
│   └── common/                # Shared utilities
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── config/                    # Environment configurations
├── k8s/                       # Kubernetes manifests
├── terraform/                 # Infrastructure as code
├── docker/                    # Dockerfiles
└── docker-compose.yaml        # Local development setup
```

## Configuration

Configuration files are located in `config/` directory:
- `development.yaml`: Local development settings
- `staging.yaml`: Staging environment settings
- `production.yaml`: Production environment settings

## Services

### Inference Service
- Port: 8000
- API Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

### Model Server
- Port: 8001
- API Docs: http://localhost:8001/docs
- Metrics: http://localhost:8001/metrics

### MLflow
- UI: http://localhost:5000

### Prometheus
- UI: http://localhost:9090

### Grafana
- UI: http://localhost:3000
- Default credentials: admin/admin

## License

MIT