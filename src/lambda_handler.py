"""
AWS Lambda handlers for fraud detection API.
Serverless deployment handlers for API Gateway integration.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy-loaded globals
_detector = None
_s3_client = None
_dynamodb = None

# Configuration
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "fraud-detection-models")
MODEL_KEY = os.getenv("MODEL_KEY", "models/fraud_model.joblib")
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "fraud-predictions")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key")


def get_detector():
    """Lazy load fraud detector with model from S3."""
    global _detector

    if _detector is None:
        import tempfile

        logger.info(f"Loading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")

        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            s3.download_file(MODEL_BUCKET, MODEL_KEY, tmp.name)

            # Import here to avoid cold start overhead if not needed
            from predict import FraudDetector

            _detector = FraudDetector(tmp.name)

        logger.info("Model loaded successfully")

    return _detector


def get_dynamodb_table():
    """Get DynamoDB table resource."""
    global _dynamodb

    if _dynamodb is None:
        dynamodb = boto3.resource("dynamodb")
        _dynamodb = dynamodb.Table(PREDICTIONS_TABLE)

    return _dynamodb


def create_response(status_code: int, body: Dict[str, Any]) -> Dict:
    """Create API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }


def log_prediction(transaction_id: str, prediction: Dict, amount: float):
    """Log prediction to DynamoDB."""
    try:
        table = get_dynamodb_table()
        timestamp = datetime.utcnow().isoformat()

        # TTL: 30 days
        ttl = int((datetime.utcnow() + timedelta(days=30)).timestamp())

        table.put_item(
            Item={
                "transaction_id": transaction_id,
                "timestamp": timestamp,
                "fraud_probability": Decimal(str(prediction["fraud_probability"])),
                "is_fraud": prediction["is_fraud"],
                "risk_level": prediction["risk_level"],
                "amount": Decimal(str(amount)),
                "ttl": ttl,
            }
        )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


# ===================
# Lambda Handlers
# ===================
def health_handler(event, context):
    """Health check endpoint."""
    return create_response(
        200,
        {
            "status": "healthy",
            "version": "1.0.0",
            "region": os.getenv("AWS_REGION", "unknown"),
        },
    )


def predict_handler(event, context):
    """
    Single transaction prediction handler.

    Expected body:
    {
        "Time": float,
        "V1": float, ..., "V28": float,
        "Amount": float
    }
    """
    try:
        body = json.loads(event.get("body", "{}"))

        # Validate required fields
        required_fields = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
        missing = [f for f in required_fields if f not in body]
        if missing:
            return create_response(
                400, {"error": "Missing required fields", "missing": missing}
            )

        # Get prediction
        detector = get_detector()
        import time

        start = time.time()
        result = detector.predict(body)
        latency_ms = (time.time() - start) * 1000

        # Generate transaction ID
        transaction_id = f"txn_{context.aws_request_id}"

        # Log to DynamoDB
        log_prediction(transaction_id, result, body["Amount"])

        return create_response(
            200,
            {
                "transaction_id": transaction_id,
                "fraud_probability": result["fraud_probability"],
                "is_fraud": result["is_fraud"],
                "risk_level": result["risk_level"],
                "threshold": result["threshold"],
                "latency_ms": round(latency_ms, 2),
            },
        )

    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return create_response(500, {"error": "Internal server error"})


def batch_predict_handler(event, context):
    """
    Batch prediction handler.

    Expected body:
    {
        "transactions": [
            {"Time": float, "V1": float, ..., "Amount": float},
            ...
        ]
    }
    """
    try:
        body = json.loads(event.get("body", "{}"))
        transactions = body.get("transactions", [])

        if not transactions:
            return create_response(400, {"error": "No transactions provided"})

        if len(transactions) > 100:
            return create_response(400, {"error": "Maximum 100 transactions per batch"})

        # Get predictions
        detector = get_detector()
        import time

        start = time.time()
        results = detector.predict_batch(transactions)
        latency_ms = (time.time() - start) * 1000

        # Format response
        predictions = []
        for i, (txn, result) in enumerate(zip(transactions, results)):
            transaction_id = f"batch_{context.aws_request_id}_{i}"
            log_prediction(transaction_id, result, txn.get("Amount", 0))

            predictions.append(
                {
                    "transaction_id": transaction_id,
                    "fraud_probability": result["fraud_probability"],
                    "is_fraud": result["is_fraud"],
                    "risk_level": result["risk_level"],
                }
            )

        return create_response(
            200,
            {
                "predictions": predictions,
                "total_transactions": len(transactions),
                "flagged_count": sum(1 for r in results if r["is_fraud"]),
                "latency_ms": round(latency_ms, 2),
            },
        )

    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return create_response(500, {"error": "Internal server error"})


def explain_handler(event, context):
    """
    Prediction with SHAP explanation handler.

    Expected body: Same as predict, with optional top_k parameter
    """
    try:
        body = json.loads(event.get("body", "{}"))
        top_k = body.pop("top_k", 10)

        # Validate required fields
        required_fields = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
        missing = [f for f in required_fields if f not in body]
        if missing:
            return create_response(
                400, {"error": "Missing required fields", "missing": missing}
            )

        # Get explanation
        detector = get_detector()
        import time

        start = time.time()
        result = detector.explain(body, top_k=top_k)
        latency_ms = (time.time() - start) * 1000

        return create_response(
            200,
            {
                "prediction": {
                    "fraud_probability": result["prediction"]["fraud_probability"],
                    "is_fraud": result["prediction"]["is_fraud"],
                    "risk_level": result["prediction"]["risk_level"],
                    "threshold": result["prediction"]["threshold"],
                    "latency_ms": round(latency_ms, 2),
                },
                "explanation": result["explanation"],
            },
        )

    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Explain error: {e}")
        return create_response(500, {"error": "Internal server error"})


def login_handler(event, context):
    """
    Authentication handler.

    Expected body:
    {
        "username": str,
        "password": str
    }
    """
    try:
        from auth import authenticate_user, create_access_token, create_refresh_token

        body = json.loads(event.get("body", "{}"))
        username = body.get("username")
        password = body.get("password")

        if not username or not password:
            return create_response(400, {"error": "Username and password required"})

        user = authenticate_user(username, password)
        if not user:
            return create_response(401, {"error": "Invalid credentials"})

        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes}
        )
        refresh_token = create_refresh_token(
            data={"sub": user.username, "scopes": user.scopes}
        )

        return create_response(
            200,
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
            },
        )

    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Login error: {e}")
        return create_response(500, {"error": "Internal server error"})


def authorizer_handler(event, context):
    """
    JWT Token Authorizer for API Gateway.

    Returns IAM policy allowing/denying access to API resources.
    """
    try:
        from auth import decode_token

        token = event.get("authorizationToken", "")
        method_arn = event.get("methodArn", "")

        # Remove 'Bearer ' prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        # Decode and validate token
        token_data = decode_token(token)

        # Generate allow policy
        return generate_policy(
            principal_id=token_data.username,
            effect="Allow",
            resource=method_arn,
            context={
                "username": token_data.username,
                "scopes": ",".join(token_data.scopes),
            },
        )

    except Exception as e:
        logger.error(f"Authorization error: {e}")
        # Return deny policy
        return generate_policy(
            principal_id="unauthorized",
            effect="Deny",
            resource=event.get("methodArn", "*"),
        )


def generate_policy(
    principal_id: str, effect: str, resource: str, context: Dict = None
) -> Dict:
    """Generate IAM policy document."""
    policy = {
        "principalId": principal_id,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {"Action": "execute-api:Invoke", "Effect": effect, "Resource": resource}
            ],
        },
    }

    if context:
        policy["context"] = context

    return policy
