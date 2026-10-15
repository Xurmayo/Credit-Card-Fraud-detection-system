import pandas as pd
import json
import time
from kafka import KafkaProducer
import os

def create_producer():
    """Create Kafka producer."""
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    return producer

def send_messages(producer, topic, data_path, delay=0.1):
    """Send messages to Kafka topic."""
    df = pd.read_csv(data_path)
    for _, row in df.iterrows():
        message = {
            'TransactionID': int(row['TransactionID']),
            'TransactionDate': row['TransactionDate'],
            'Amount': float(row['Amount']),
            'MerchantID': int(row['MerchantID']),
            'TransactionType': row['TransactionType'],
            'Location': row['Location'],
            'IsFraud': int(row['IsFraud'])
        }
        producer.send(topic, value=message)
        print(f"Sent: {message}")
        time.sleep(delay)  # Simulate real-time

    producer.flush()

if __name__ == "__main__":
    topic = 'fraud_detection'
    data_path = '../data/credit_card_fraud_dataset.csv'

    producer = create_producer()
    send_messages(producer, topic, data_path)
    producer.close()
    print("All messages sent.")