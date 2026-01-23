"""
Kafka consumer for real-time fraud detection streaming pipeline.
Consumes transactions from Kafka topic and produces fraud predictions.
"""
import os
import json
import logging
from typing import Optional
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from predict import FraudDetector
from drift import get_drift_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_INPUT_TOPIC = os.getenv('KAFKA_INPUT_TOPIC', 'transactions')
KAFKA_OUTPUT_TOPIC = os.getenv('KAFKA_OUTPUT_TOPIC', 'fraud-predictions')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'fraud-detector')
KAFKA_AUTO_OFFSET_RESET = os.getenv('KAFKA_AUTO_OFFSET_RESET', 'latest')


class FraudDetectionConsumer:
    """
    Kafka consumer that processes transactions and produces fraud predictions.

    Architecture:
    1. Consume transaction from input topic
    2. Run fraud detection model
    3. Produce prediction to output topic
    4. Track metrics and drift
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        input_topic: str = KAFKA_INPUT_TOPIC,
        output_topic: str = KAFKA_OUTPUT_TOPIC,
        group_id: str = KAFKA_GROUP_ID,
        model_path: str = "models/fraud_model.joblib"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id

        # Initialize fraud detector
        logger.info(f"Loading fraud detection model from {model_path}")
        self.detector = FraudDetector(model_path)
        logger.info("Model loaded successfully")

        # Initialize drift detector
        self.drift_detector = get_drift_detector()

        # Kafka consumer
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None

        # Statistics
        self.processed_count = 0
        self.fraud_count = 0
        self.error_count = 0

    def _create_consumer(self) -> KafkaConsumer:
        """Create Kafka consumer instance."""
        return KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None
        )

    def _create_producer(self) -> KafkaProducer:
        """Create Kafka producer instance."""
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            acks='all',
            retries=3
        )

    def process_transaction(self, transaction: dict, transaction_id: str = None) -> dict:
        """
        Process a single transaction and return prediction result.

        Args:
            transaction: Transaction data with V1-V28, Amount, Time
            transaction_id: Optional transaction identifier

        Returns:
            Prediction result with fraud probability and risk level
        """
        try:
            # Run prediction
            prediction = self.detector.predict(transaction)

            # Add transaction to drift detector
            feature_data = {k: v for k, v in transaction.items() if k.startswith('V') or k in ['Amount', 'Time']}
            self.drift_detector.add_sample(feature_data)

            # Build result
            result = {
                'transaction_id': transaction_id or f"txn_{self.processed_count}",
                'timestamp': datetime.utcnow().isoformat(),
                'fraud_probability': prediction['fraud_probability'],
                'is_fraud': prediction['is_fraud'],
                'risk_level': prediction['risk_level'],
                'threshold': prediction['threshold'],
                'amount': transaction.get('Amount', 0),
                'processing_status': 'success'
            }

            # Update statistics
            self.processed_count += 1
            if prediction['is_fraud']:
                self.fraud_count += 1

            return result

        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            self.error_count += 1
            return {
                'transaction_id': transaction_id,
                'timestamp': datetime.utcnow().isoformat(),
                'processing_status': 'error',
                'error': str(e)
            }

    def run(self):
        """
        Main consumer loop. Continuously processes transactions from Kafka.
        """
        logger.info(f"Starting Kafka consumer...")
        logger.info(f"  Bootstrap servers: {self.bootstrap_servers}")
        logger.info(f"  Input topic: {self.input_topic}")
        logger.info(f"  Output topic: {self.output_topic}")
        logger.info(f"  Consumer group: {self.group_id}")

        try:
            self.consumer = self._create_consumer()
            self.producer = self._create_producer()

            logger.info("Consumer started. Waiting for messages...")

            for message in self.consumer:
                try:
                    transaction = message.value
                    transaction_id = message.key

                    logger.debug(f"Received transaction: {transaction_id}")

                    # Process transaction
                    result = self.process_transaction(transaction, transaction_id)

                    # Produce result to output topic
                    self.producer.send(
                        self.output_topic,
                        key=result['transaction_id'],
                        value=result
                    )

                    # Log progress periodically
                    if self.processed_count % 100 == 0:
                        fraud_rate = self.fraud_count / self.processed_count if self.processed_count > 0 else 0
                        logger.info(
                            f"Processed: {self.processed_count} | "
                            f"Fraud: {self.fraud_count} ({fraud_rate:.2%}) | "
                            f"Errors: {self.error_count}"
                        )

                        # Check for drift periodically
                        drift_report = self.drift_detector.detect_drift(min_samples=100)
                        if drift_report and drift_report.drift_detected:
                            logger.warning(
                                f"DRIFT DETECTED! Score: {drift_report.overall_drift_score:.3f} | "
                                f"Features affected: {drift_report.features_with_drift}/{drift_report.total_features}"
                            )

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            raise
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer closed")
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed")

    def get_stats(self) -> dict:
        """Get processing statistics."""
        fraud_rate = self.fraud_count / self.processed_count if self.processed_count > 0 else 0
        return {
            'processed_count': self.processed_count,
            'fraud_count': self.fraud_count,
            'error_count': self.error_count,
            'fraud_rate': fraud_rate
        }


def main():
    """Entry point for Kafka consumer."""
    consumer = FraudDetectionConsumer()
    try:
        consumer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stats = consumer.get_stats()
        logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
