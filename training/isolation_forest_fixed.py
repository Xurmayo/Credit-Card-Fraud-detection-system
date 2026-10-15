import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_isolation_forest(X_train, contamination=0.1):
    """Train Isolation Forest model."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    # Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = model.predict(X_test)
    # Convert to 0/1 (1 for anomaly, 0 for normal)
    predictions = np.where(predictions == -1, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Isolation Forest Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Isolation Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../models/isolation_forest_confusion_matrix.png')
    plt.close()

    # Anomaly Scores
    scores = model.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, label='Scores')
    plt.axvline(x=0, color='red', linestyle='--', label='Threshold')
    plt.title('Anomaly Score Distribution - Isolation Forest')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('../models/isolation_forest_scores.png')
    plt.close()

    return accuracy, precision, recall, f1

def save_model(model, filepath):
    """Save the trained model."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Load processed data
    data_dir = '../data/'
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Train model
    contamination = 0.1  # Adjust based on dataset
    model = train_isolation_forest(X_train, contamination)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, '../models/isolation_forest.pkl')
