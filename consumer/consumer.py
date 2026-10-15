import json
import numpy as np
import joblib
from kafka import KafkaConsumer
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """Load trained models and preprocessors."""
    scaler = joblib.load('../models/scaler.pkl')
    le_type = joblib.load('../models/le_type.pkl')
    le_location = joblib.load('../models/le_location.pkl')
    iso_forest = joblib.load('../models/isolation_forest.pkl')
    autoencoder = joblib.load('../models/autoencoder.pkl')
    return scaler, le_type, le_location, iso_forest, autoencoder

def preprocess_message(message, scaler, le_type, le_location):
    """Preprocess incoming message."""
    # Extract features
    features = [
        message['Amount'],
        message['MerchantID'],
        le_type.transform([message['TransactionType']])[0],
        le_location.transform([message['Location']])[0]
    ]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    return features_scaled

def predict_anomaly(features, iso_forest, autoencoder):
    """Predict anomaly using both models."""
    # Isolation Forest
    iso_pred = iso_forest.predict(features)[0]
    iso_score = iso_forest.decision_function(features)[0]
    iso_anomaly = 1 if iso_pred == -1 else 0

    # Autoencoder
    reconstruction = autoencoder.predict(features)
    ae_score = np.mean(np.power(features - reconstruction, 2))
    threshold = 0.1  # This should be set based on training, but for simplicity
    ae_anomaly = 1 if ae_score > threshold else 0

    return iso_anomaly, iso_score, ae_anomaly, ae_score

def consume_messages(topic, scaler, le_type, le_location, iso_forest, autoencoder):
    """Consume messages and process them."""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    latencies = []
    for message in consumer:
        start_time = time.time()

        data = message.value
        logger.info(f"Received: {data}")

        # Preprocess
        features = preprocess_message(data, scaler, le_type, le_location)

        # Predict
        iso_anomaly, iso_score, ae_anomaly, ae_score = predict_anomaly(features, iso_forest, autoencoder)

        # Log results
        result = {
            'TransactionID': data['TransactionID'],
            'TrueLabel': data['IsFraud'],
            'IsoForest_Anomaly': iso_anomaly,
            'IsoForest_Score': iso_score,
            'Autoencoder_Anomaly': ae_anomaly,
            'Autoencoder_Score': ae_score
        }
        logger.info(f"Prediction: {result}")

        # Measure latency
        latency = time.time() - start_time
        latencies.append(latency)
        if len(latencies) % 100 == 0:
            avg_latency = np.mean(latencies[-100:])
            throughput = 1 / avg_latency if avg_latency > 0 else 0
            logger.info(f"Average latency: {avg_latency:.4f}s, Throughput: {throughput:.2f} msg/s")

if __name__ == "__main__":
    topic = 'fraud_detection'

    # Load models
    scaler, le_type, le_location, iso_forest, autoencoder = load_models()

    # Start consuming
    consume_messages(topic, scaler, le_type, le_location, iso_forest, autoencoder)