# Fraud Detection Pipeline

Real-time credit card fraud detection API. Built with XGBoost on the Kaggle credit card dataset (284K transactions, 0.17% fraud rate).

**[Try the Live Demo](https://huggingface.co/spaces/AB-316/fraud-detection)**

## Results

- **94% recall** at 5% false positive rate
- **PR-AUC: 0.85**
- **<100ms inference latency**

The model is tuned for high recall because missing fraud is more expensive than flagging legitimate transactions for review.

## What's in here

```
src/
  features.py      - feature engineering, SMOTE balancing
  train.py         - model training with MLflow tracking
  predict.py       - inference with SHAP explanations
  drift.py         - monitors feature distribution shifts
  kafka_consumer.py - streaming predictions

api/app.py         - FastAPI with auth, rate limiting, metrics
dashboard/app.py   - Streamlit monitoring UI
```

## Run it locally

```bash
# setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# get the data from kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
mv ~/Downloads/creditcard.csv data/

# train
python src/train.py

# run api
python api/app.py
# http://localhost:8000/docs
```

## Docker

```bash
# just the api + dashboard
docker-compose up -d

# with prometheus/grafana monitoring
docker-compose --profile monitoring up -d
```

## API

Login first to get a token:
```bash
curl -X POST localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Then predict:
```bash
curl -X POST localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "Amount": 149.62, "V1": -1.36, "V2": -0.07, ...}'
```

The `/predict/explain` endpoint returns SHAP values showing why a transaction was flagged.

## Tech

- XGBoost with SMOTE for class imbalance
- SHAP for explainability (regulatory requirement in fintech)
- FastAPI with JWT auth and rate limiting
- Prometheus metrics + Grafana dashboards
- Kafka consumer for streaming
- MLflow for experiment tracking
- Docker + GitHub Actions CI

## Why these choices

**XGBoost over deep learning** - Faster inference, interpretable feature importance, handles tabular data well. No need for neural nets here.

**94% recall target** - In fraud detection, false negatives (missed fraud) cost way more than false positives (extra manual reviews). Tuned the threshold accordingly.

**SHAP** - Regulators want to know why transactions get flagged. Can't deploy a black box in fintech.
