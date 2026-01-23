# Fraud Detection Pipeline

Production-ready real-time credit card fraud detection system using XGBoost, featuring explainable AI, streaming predictions, and comprehensive monitoring.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Model Performance

| Metric | Value |
|--------|-------|
| Recall | 94% |
| Precision | ~13% |
| False Positive Rate | 5% |
| PR AUC | 0.85 |
| Inference Latency | <100ms |

## Features

### Core ML
- **XGBoost classifier** optimized for imbalanced fraud detection
- **SMOTE oversampling** to handle 0.17% fraud rate
- **Feature engineering** with time-based and interaction features
- **Threshold tuning** for configurable recall/precision tradeoff

### Explainability
- **SHAP integration** for model interpretability
- Per-prediction feature contributions
- Regulatory-compliant explanations

### API & Security
- **FastAPI** with async support
- **JWT authentication** with role-based access control
- **Rate limiting** per endpoint and user
- OpenAPI/Swagger documentation

### Monitoring & Observability
- **Prometheus metrics** (latency, throughput, fraud rate)
- **Grafana dashboards** pre-configured
- **Feature drift detection** using KS-test and PSI

### MLOps
- **MLflow integration** for experiment tracking
- Model versioning and registry
- Automated training pipeline

### Streaming
- **Kafka consumer** for real-time transaction processing
- Scalable event-driven architecture

### Infrastructure
- **Docker & Docker Compose** with multi-stage builds
- **AWS SAM template** for serverless deployment
- **GitHub Actions CI/CD** pipeline

## Project Structure

```
fraud-detection-pipeline/
├── src/
│   ├── features.py         # Feature engineering + SMOTE
│   ├── train.py            # Model training + MLflow
│   ├── predict.py          # Inference + SHAP explanations
│   ├── auth.py             # JWT authentication
│   ├── metrics.py          # Prometheus metrics
│   ├── drift.py            # Feature drift detection
│   ├── kafka_consumer.py   # Streaming consumer
│   └── lambda_handler.py   # AWS Lambda handlers
├── api/
│   └── app.py              # FastAPI service
├── dashboard/
│   └── app.py              # Streamlit monitoring UI
├── tests/
│   ├── test_features.py    # Feature engineering tests
│   ├── test_predict.py     # Prediction tests
│   └── test_api.py         # API endpoint tests
├── infrastructure/
│   ├── template.yaml       # AWS SAM template
│   ├── prometheus.yml      # Prometheus config
│   └── grafana/            # Grafana dashboards
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)

### Local Development

```bash
# Clone repository
git clone https://github.com/AnthonyB-316/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
mv ~/Downloads/creditcard.csv data/

# Train model
python src/train.py

# Run API
python api/app.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Docker Deployment

```bash
# Core services (API + Dashboard)
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# With streaming (Kafka)
docker-compose --profile streaming up -d

# All services
docker-compose --profile monitoring --profile streaming up -d
```

**Service URLs:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- MLflow: http://localhost:5000

## API Usage

### Authentication

```bash
# Get JWT token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Demo credentials:
# admin / admin123 (full access)
# analyst / analyst123 (read only)
# api_user / api123 (read + write)
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.0234,
  "is_fraud": false,
  "risk_level": "LOW",
  "threshold": 0.15,
  "latency_ms": 12.34
}
```

### Prediction with Explanation

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{ ... transaction ... }'
```

**Response includes SHAP explanation:**
```json
{
  "prediction": { ... },
  "explanation": {
    "base_value": 0.12,
    "top_features": [
      {"feature": "V14", "contribution": -0.08, "direction": "decreases fraud risk"},
      {"feature": "V4", "contribution": 0.05, "direction": "increases fraud risk"}
    ]
  }
}
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | - | Health check |
| `/metrics` | GET | - | Prometheus metrics |
| `/auth/login` | POST | - | Get JWT token |
| `/auth/me` | GET | Yes | Current user info |
| `/predict` | POST | read | Single prediction |
| `/predict/batch` | POST | write | Batch predictions (max 100) |
| `/predict/explain` | POST | read | Prediction + SHAP explanation |
| `/model/info` | GET | read | Model metrics & feature importance |
| `/drift/status` | GET | read | Feature drift report |
| `/drift/stats` | GET | read | Production data statistics |

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_predictions_total` | Counter | Total predictions by result/risk |
| `fraud_prediction_latency_seconds` | Histogram | Prediction latency |
| `fraud_prediction_probability` | Histogram | Probability distribution |
| `feature_drift_score` | Gauge | Drift score per feature |
| `http_requests_total` | Counter | HTTP requests by endpoint |

### Grafana Dashboard

Pre-configured dashboard includes:
- Predictions per second
- Fraud rate over time
- Latency percentiles (p50, p95, p99)
- Risk level distribution
- Feature drift scores

## AWS Deployment

### SAM Deployment

```bash
cd infrastructure

# Build
sam build

# Deploy
sam deploy --guided \
  --parameter-overrides \
    Environment=prod \
    ModelBucketName=your-model-bucket \
    JwtSecretKey=your-secret-key
```

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ API Gateway │────▶│   Lambda    │────▶│  DynamoDB   │
└─────────────┘     │  (Predict)  │     │ (Predictions)│
                    └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     S3      │
                    │   (Model)   │
                    └─────────────┘
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py -v
```

## MLflow

```bash
# View experiments (if running locally)
mlflow ui

# Train with MLflow tracking
python src/train.py

# Train without MLflow
python src/train.py --no-mlflow
```

## Key Design Decisions

### Why XGBoost?
- Handles imbalanced data well with threshold tuning
- Fast inference (<100ms) for real-time predictions
- Native feature importance for interpretability
- Production-proven in fintech applications

### Why SMOTE?
- Dataset is extremely imbalanced (0.17% fraud)
- Oversamples minority class to improve recall
- Applied only to training data to prevent leakage

### Why 94% Recall Target?
- In fraud detection, false negatives (missed fraud) are costly
- Accepting higher false positive rate for manual review
- Configurable based on business requirements

### Why SHAP?
- Regulatory requirement for explainable AI in finance
- Per-prediction explanations for customer disputes
- Feature importance for model debugging

## Author

Anthony Buonantuono - [GitHub](https://github.com/AnthonyB-316)

## License

MIT License
