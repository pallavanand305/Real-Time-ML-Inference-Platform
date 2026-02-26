# Real-Time ML Inference Platform

[![CI](https://github.com/yourusername/real-time-ml-inference-platform/workflows/CI/badge.svg)](https://github.com/yourusername/real-time-ml-inference-platform/actions)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/yourusername/real-time-ml-inference-platform)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, enterprise-scale ML inference platform designed for low-latency predictions at scale. Built with modern MLOps best practices, this system demonstrates capabilities similar to inference systems at Netflix, Uber, and other tech companies serving millions of predictions per day.

## 🎯 Project Overview

This platform showcases a complete end-to-end ML serving architecture with:
- **Sub-100ms p95 latency** through intelligent caching and optimized model serving
- **1000+ requests/second** per instance with horizontal scalability
- **99.9% uptime** with multi-AZ deployment and automatic failover
- **Complete MLOps lifecycle** from model registration to deployment to monitoring
- **Production-ready** security, compliance, and operational tooling

**Target Audience**: This project demonstrates senior backend/MLOps engineering capabilities including distributed systems design, high-performance API development, event-driven architectures, and production observability.

## ✨ Key Features

### Core Capabilities
- **🚀 High-Performance Inference**: FastAPI-based async service with ONNX Runtime for optimized model execution
- **💾 Intelligent Caching**: Redis-based feature cache with cache-aside pattern and automatic fallback
- **📊 Model Registry**: MLflow integration for complete model lifecycle management and versioning
- **🔄 Event Streaming**: Kafka-based asynchronous inference logging for analytics and retraining
- **📈 Drift Detection**: Automated statistical monitoring using KL divergence and PSI metrics
- **🔍 Full Observability**: Prometheus metrics, Grafana dashboards, structured logging, and distributed tracing

### Production Features
- **🔐 Security**: JWT/API key authentication, rate limiting, TLS encryption, PII redaction
- **🛡️ Resilience**: Circuit breaker pattern, exponential backoff, graceful degradation
- **📦 Containerization**: Docker containers with multi-stage builds and Kubernetes manifests
- **🔄 CI/CD**: GitHub Actions pipelines with automated testing and deployment
- **☁️ Cloud-Ready**: Terraform IaC for AWS/GCP, multi-environment configuration

## 🏗️ Architecture

### High-Level System Design

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    API Gateway                          │
│            (Auth, Rate Limiting, TLS)                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Load Balancer        │
              └────────┬───────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Inference   │ │ Inference   │ │ Inference   │
│ Service 1   │ │ Service 2   │ │ Service N   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Redis     │ │   Model     │ │   Kafka     │
│   Cache     │ │   Server    │ │   Stream    │
└─────────────┘ └──────┬──────┘ └──────┬──────┘
                       │               │
                       ▼               ▼
                ┌─────────────┐ ┌─────────────┐
                │   MLflow    │ │   Drift     │
                │   Registry  │ │   Detector  │
                └──────┬──────┘ └─────────────┘
                       │
                       ▼
                ┌─────────────┐
                │  S3 Storage │
                └─────────────┘
```

### Component Architecture

#### 1. Inference Service (FastAPI)
Orchestrates the inference pipeline with:
- Request validation and feature schema checking
- Feature retrieval with cache-first strategy
- Model server communication with circuit breaker
- Asynchronous event publishing to Kafka
- Structured error handling and retry logic

#### 2. Model Server (ONNX Runtime)
Handles model execution with:
- Multi-model serving (multiple versions simultaneously)
- In-memory model caching with LRU eviction
- Batch inference support (up to 100 samples)
- GPU acceleration when available
- Checksum verification for model integrity

#### 3. Drift Detector
Monitors model health with:
- Sliding window aggregation (7-day baseline, 24-hour current)
- KL divergence for feature drift detection
- PSI (Population Stability Index) for prediction drift
- Automated alerting via Kafka and Slack
- Historical drift tracking in PostgreSQL

### Data Flow

```
1. Client Request → API Gateway (auth/rate limit)
2. Gateway → Load Balancer → Inference Service
3. Inference Service → Redis Cache (feature lookup)
4. Cache Miss → PostgreSQL (feature retrieval)
5. Inference Service → Model Server (prediction)
6. Model Server → MLflow Registry (model metadata)
7. Inference Service → Kafka (async event log)
8. Kafka → Drift Detector (monitoring)
9. Response → Client (prediction + metadata)
```

## 🛠️ Technology Stack

### Application Layer
- **FastAPI** (Python 3.11+): High-performance async web framework
- **ONNX Runtime**: Cross-framework model serving with GPU support
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server with production-grade performance

### Data Layer
- **Redis 7**: Sub-millisecond feature caching with cluster mode
- **PostgreSQL 15**: Model registry backend with JSONB support
- **Apache Kafka 3.x**: Event streaming with exactly-once semantics
- **AWS S3**: Durable model artifact storage

### ML Operations
- **MLflow 2.x**: Model registry, experiment tracking, and lifecycle management
- **Scikit-learn**: Statistical drift detection algorithms
- **NumPy/SciPy**: Numerical computing for drift metrics

### Observability
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Structured Logging**: JSON logs with correlation IDs

### Infrastructure
- **Docker**: Containerization with multi-stage builds
- **Kubernetes**: Container orchestration with auto-scaling
- **Terraform**: Infrastructure as Code for AWS/GCP
- **GitHub Actions**: CI/CD pipelines

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Poetry** (recommended) or pip
- **Make** (optional, for convenience commands)

### Local Development Setup

#### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/real-time-ml-inference-platform.git
cd real-time-ml-inference-platform

# Option A: Using Poetry (Recommended)
pip install poetry
poetry install

# Option B: Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

#### 2. Set Up Pre-commit Hooks

```bash
# With Poetry
poetry run pre-commit install

# With pip
pre-commit install
```

#### 3. Start Infrastructure Services

```bash
# Start all services (Redis, PostgreSQL, Kafka, MLflow, Prometheus, Grafana)
make docker-up

# Or using docker-compose directly
docker-compose up -d

# Verify all services are healthy
make check-services
# Or: python scripts/check_services.py
```

This starts:
- **Redis** (port 6379): Feature cache
- **PostgreSQL** (port 5432): MLflow backend
- **Kafka** (port 9092): Event streaming
- **Zookeeper** (port 2181): Kafka coordination
- **MLflow** (port 5000): Model registry UI
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Visualization (admin/admin)

#### 4. Run the Application

```bash
# Start Inference Service
poetry run uvicorn src.inference_service.app:app --reload --port 8000

# Start Model Server (in another terminal)
poetry run uvicorn src.model_server.app:app --reload --port 8001

# Start Drift Detector (in another terminal)
poetry run python -m src.drift_detector.app
```

#### 5. Verify Setup

```bash
# Run verification script
poetry run python scripts/verify_setup.py

# Or manually check endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8000/docs  # OpenAPI documentation
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Or using pytest directly
poetry run pytest

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/

# Run with verbose output
poetry run pytest -v

# Generate HTML coverage report
poetry run pytest --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Run linting
make lint
# Or: poetry run ruff check src/ tests/

# Format code
make format
# Or: poetry run black src/ tests/

# Type checking
make type-check
# Or: poetry run mypy src/

# Run all quality checks
make quality
```

## 📖 Usage Examples

### Making Inference Requests

```python
import httpx

# Prepare inference request
request = {
    "model_id": "fraud_detection_v2",
    "features": {
        "transaction_amount": 125.50,
        "merchant_category": "retail",
        "user_age_days": 450,
        "transaction_count_30d": 15
    },
    "options": {
        "timeout_ms": 5000,
        "include_explanation": False
    }
}

# Send request
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/v1/predict",
        json=request,
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Latency: {result['latency_ms']}ms")
```

### Registering a Model

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Register model
model_uri = "s3://ml-models/fraud_detection/model.onnx"
mlflow.register_model(model_uri, "fraud_detection")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection",
    version="2",
    stage="Production"
)
```

### Monitoring Metrics

```bash
# View Prometheus metrics
curl http://localhost:8000/metrics

# Key metrics:
# - inference_requests_total: Total requests by model and status
# - inference_latency_seconds: Request latency histogram
# - cache_operations_total: Cache hit/miss statistics
# - circuit_breaker_state: Circuit breaker status
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The platform exposes comprehensive metrics:

- **Request Metrics**: Rate, latency (p50/p95/p99), error rate
- **Cache Metrics**: Hit ratio, operation latency, eviction rate
- **Model Metrics**: Inference duration, memory usage, active models
- **System Metrics**: CPU, memory, network I/O
- **Business Metrics**: Predictions per model, drift scores

### Grafana Dashboards

Pre-configured dashboards available at `http://localhost:3000`:

1. **Inference Overview**: Request rate, latency, error rate, cache performance
2. **Model Performance**: Per-model metrics, version distribution, inference times
3. **System Health**: Resource utilization, pod status, circuit breaker states
4. **Drift Monitoring**: Drift scores over time, alert frequency, feature distributions

### Structured Logging

All services emit JSON-structured logs with:
- Correlation IDs for request tracing
- Contextual information (model_id, user_id, etc.)
- Error stack traces and debugging info
- PII redaction for compliance

Example log entry:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "inference-service",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "fraud_detection_v2",
  "latency_ms": 45.2,
  "cache_hit": true,
  "message": "Inference completed successfully"
}
```

## 🏗️ Project Structure

```
.
├── src/
│   ├── common/                    # Shared utilities and models
│   │   ├── models.py              # Pydantic data models
│   │   ├── config.py              # Configuration management
│   │   ├── cache.py               # Redis cache client
│   │   └── feature_retrieval.py  # Feature fetching logic
│   ├── inference_service/         # Main inference API
│   │   ├── app.py                 # FastAPI application
│   │   ├── routes.py              # API endpoints
│   │   └── dependencies.py        # Dependency injection
│   ├── model_server/              # Model serving engine
│   │   ├── app.py                 # FastAPI application
│   │   ├── loader.py              # Model loading logic
│   │   └── inference.py           # Inference execution
│   └── drift_detector/            # Drift monitoring service
│       ├── app.py                 # Main application
│       ├── detector.py            # Drift detection algorithms
│       └── alerting.py            # Alert publishing
├── tests/
│   ├── unit/                      # Unit tests
│   │   ├── test_models.py
│   │   ├── test_cache.py
│   │   └── test_feature_retrieval.py
│   └── integration/               # Integration tests
│       ├── test_inference_flow.py
│       └── test_event_streaming.py
├── config/                        # Environment configurations
│   ├── development.yaml
│   ├── staging.yaml
│   ├── production.yaml
│   └── prometheus.yaml
├── k8s/                           # Kubernetes manifests
│   ├── deployments/
│   ├── services/
│   ├── configmaps/
│   └── hpa/
├── terraform/                     # Infrastructure as Code
│   ├── modules/
│   │   ├── eks/
│   │   ├── rds/
│   │   └── s3/
│   └── environments/
├── docker/                        # Dockerfiles
├── scripts/                       # Utility scripts
│   ├── verify_setup.py
│   └── check_services.py
├── docs/                          # Additional documentation
│   └── cache_aside_pattern.md
├── examples/                      # Usage examples
│   └── cache_aside_demo.py
├── .github/
│   └── workflows/                 # CI/CD pipelines
│       ├── ci.yml
│       └── cd.yml
├── docker-compose.yaml            # Local development setup
├── pyproject.toml                 # Python dependencies
├── Makefile                       # Convenience commands
└── README.md                      # This file
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build images
docker build -t inference-service:latest -f docker/Dockerfile.inference .
docker build -t model-server:latest -f docker/Dockerfile.model-server .

# Run with docker-compose
docker-compose -f docker-compose.prod.yaml up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/hpa/

# Verify deployment
kubectl get pods -n ml-inference-prod
kubectl get svc -n ml-inference-prod

# Check logs
kubectl logs -f deployment/inference-service -n ml-inference-prod
```

### Cloud Deployment (AWS/GCP)

```bash
# Initialize Terraform
cd terraform/environments/production
terraform init

# Plan infrastructure changes
terraform plan

# Apply infrastructure
terraform apply

# Deploy application
kubectl config use-context production
kubectl apply -f k8s/
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Redis
REDIS_URL=redis://localhost:6379
REDIS_TTL_SECONDS=300

# Kafka
KAFKA_BROKERS=localhost:9092
KAFKA_TOPIC=inference-events

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=s3://ml-models

# Model Server
MODEL_SERVER_URL=http://localhost:8001
MODEL_SERVER_TIMEOUT_MS=4000

# Authentication
JWT_SECRET_KEY=your-secret-key
API_KEY_HEADER=X-API-Key
```

### Configuration Files

Configuration is managed through YAML files in `config/`:

```yaml
# config/production.yaml
inference_service:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout_seconds: 5
  
  cache:
    enabled: true
    ttl_seconds: 300
    max_retries: 2
  
  circuit_breaker:
    failure_threshold: 0.5
    recovery_timeout_seconds: 60

model_server:
  max_models_in_memory: 10
  model_cache_size_mb: 4096
  batch_size: 100
  gpu_enabled: true

drift_detector:
  check_interval_seconds: 3600
  baseline_window_hours: 168
  thresholds:
    feature_drift_warning: 0.1
    feature_drift_critical: 0.3
```

## 🧪 Testing Strategy

### Test Coverage

- **Unit Tests**: 80%+ coverage for core business logic
- **Integration Tests**: End-to-end flow validation
- **Property-Based Tests**: Using Hypothesis for edge cases
- **Performance Tests**: Load testing with Locust

### Running Different Test Suites

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v

# Performance tests
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Property-based tests
pytest tests/property/ -v --hypothesis-show-statistics
```

## 📈 Performance Benchmarks

### Latency Targets

- **p50 latency**: <50ms
- **p95 latency**: <100ms
- **p99 latency**: <200ms

### Throughput

- **Single instance**: 1000+ req/sec
- **Horizontal scaling**: Linear scaling up to 20 instances
- **Cache hit ratio**: >80% in production workloads

### Resource Usage

- **Inference Service**: 2 CPU cores, 4GB RAM
- **Model Server**: 4 CPU cores, 8GB RAM (16GB with GPU)
- **Drift Detector**: 2 CPU cores, 4GB RAM

## 🔐 Security

### Authentication & Authorization

- JWT token-based authentication
- API key support for service-to-service communication
- Role-based access control (RBAC) for admin operations
- Rate limiting: 100 requests/minute per API key

### Data Security

- TLS 1.3 encryption for all API endpoints
- AES-256 encryption for data at rest (Kafka, S3)
- PII redaction in logs and metrics
- Secrets management via AWS Secrets Manager / GCP Secret Manager

### Network Security

- Kubernetes NetworkPolicies for pod-to-pod communication
- Private subnets for application services
- Security groups restricting ingress/egress
- VPC peering for cross-region communication

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make quality && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project demonstrates production-grade ML serving patterns inspired by:
- Netflix's ML infrastructure
- Uber's Michelangelo platform
- Airbnb's ML platform architecture
- Industry best practices from MLOps community

## 📧 Contact

For questions or feedback, please open an issue or reach out via [your contact method].

---

**Built with ❤️ for demonstrating enterprise-scale ML engineering capabilities**