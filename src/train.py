"""
Train fraud detection model using XGBoost with MLflow tracking.
Optimized for high recall while maintaining acceptable precision.
"""
import os
import joblib
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score
)
import matplotlib.pyplot as plt
import logging

from features import load_data, engineer_features, prepare_train_test

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "fraud-detection"


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")


def train_model(X_train, y_train, params=None, random_state=42):
    """
    Train XGBoost classifier optimized for fraud detection.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Optional dict of XGBoost parameters
        random_state: Random seed for reproducibility

    Returns:
        Trained XGBClassifier model
    """
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,  # Already balanced via SMOTE
        'random_state': random_state,
        'eval_metric': 'aucpr',
        'use_label_encoder': False
    }

    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model with focus on fraud detection metrics.

    Returns:
        Dictionary of evaluation metrics
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("=" * 50)
    print(f"EVALUATION (threshold={threshold})")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\nKey Metrics:")
    print(f"  Recall (Fraud caught): {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")

    return {
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def find_optimal_threshold(model, X_test, y_test, target_recall=0.90):
    """Find threshold that achieves target recall."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Find threshold for target recall
    for i, recall in enumerate(recalls):
        if recall <= target_recall and i > 0:
            optimal_threshold = thresholds[i - 1]
            optimal_precision = precisions[i - 1]
            print(f"\nOptimal threshold for {target_recall:.0%} recall: {optimal_threshold:.4f}")
            print(f"  Precision at this threshold: {optimal_precision:.4f}")
            return optimal_threshold

    return 0.5


def plot_metrics(model, X_test, y_test, save_path="models/metrics.png"):
    """Plot precision-recall curve and feature importance."""
    y_proba = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision-Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve (AUC={average_precision_score(y_test, y_proba):.3f})')
    axes[0].grid(True, alpha=0.3)

    # Feature Importance (top 15)
    importance = model.feature_importances_
    feature_names = model.feature_names_in_
    indices = np.argsort(importance)[-15:]
    axes[1].barh(range(len(indices)), importance[indices])
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([feature_names[i] for i in indices])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Features')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nMetrics plot saved to {save_path}")

    return fig


def main(use_mlflow=True, target_recall=0.94):
    """
    Main training pipeline.

    Args:
        use_mlflow: Whether to log to MLflow
        target_recall: Target recall for threshold tuning
    """
    if use_mlflow:
        setup_mlflow()

    # Start MLflow run
    with mlflow.start_run() if use_mlflow else nullcontext():
        print("Loading and preparing data...")
        df = load_data("data/creditcard.csv")
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)

        # Log data parameters
        if use_mlflow:
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("target_recall", target_recall)
            mlflow.log_param("fraud_rate_original", df['Class'].mean())

        # Model parameters
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        if use_mlflow:
            mlflow.log_params(params)

        print("\nTraining model...")
        model = train_model(X_train, y_train, params=params)

        print("\nEvaluating at default threshold (0.5)...")
        default_metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

        if use_mlflow:
            mlflow.log_metrics({f"default_{k}": v for k, v in default_metrics.items()
                             if isinstance(v, (int, float))})

        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(model, X_test, y_test, target_recall=target_recall)

        print("\nEvaluating at optimal threshold...")
        metrics = evaluate_model(model, X_test, y_test, threshold=optimal_threshold)

        if use_mlflow:
            mlflow.log_param("optimal_threshold", optimal_threshold)
            mlflow.log_metrics({f"optimal_{k}": v for k, v in metrics.items()
                             if isinstance(v, (int, float))})

        # Save model
        print("\nSaving model...")
        os.makedirs("models", exist_ok=True)

        model_artifact = {
            'model': model,
            'threshold': optimal_threshold,
            'feature_cols': feature_cols,
            'metrics': metrics
        }
        joblib.dump(model_artifact, "models/fraud_model.joblib")
        print("Model saved to models/fraud_model.joblib")

        # Log model to MLflow
        if use_mlflow:
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name="fraud-detection-xgboost"
            )
            mlflow.log_artifact("models/fraud_model.joblib")

        # Generate and save plots
        fig = plot_metrics(model, X_test, y_test)
        if use_mlflow:
            mlflow.log_figure(fig, "metrics.png")
            mlflow.log_artifact("models/metrics.png")

        print("\n" + "=" * 50)
        print("TRAINING COMPLETE")
        print("=" * 50)
        print(f"Model: XGBoost")
        print(f"Threshold: {optimal_threshold:.4f}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")

        if use_mlflow:
            run_id = mlflow.active_run().info.run_id
            print(f"\nMLflow Run ID: {run_id}")
            print(f"View at: {MLFLOW_TRACKING_URI}")

        return model, metrics


# Context manager for optional MLflow
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--target-recall", type=float, default=0.94, help="Target recall for threshold")
    args = parser.parse_args()

    main(use_mlflow=not args.no_mlflow, target_recall=args.target_recall)
