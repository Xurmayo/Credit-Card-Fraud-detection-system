import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_isolation_forest(X_train, contamination=0.1, random_state=42):
    """Train an Isolation Forest anomaly detection model."""
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=50,
        max_samples=256,
        max_features=0.8,
        n_jobs=1,
        verbose=0
    )
    model.fit(X_train)
    return model


def predict_scores(model, X):
    """Predict anomaly scores and binary labels."""
    raw_scores = model.decision_function(X)
    anomaly_scores = -raw_scores
    predictions = np.where(model.predict(X) == -1, 1, 0)
    return anomaly_scores, predictions


def plot_roc_pr_curves(y_true, y_scores, name_prefix):
    """Plot ROC and Precision-Recall curves for a model."""
    os.makedirs('../models', exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name_prefix}')
    plt.legend()
    plt.savefig(f'../models/{name_prefix}_roc_curve.png')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name_prefix}')
    plt.legend()
    plt.savefig(f'../models/{name_prefix}_pr_curve.png')
    plt.close()

    return roc_auc, ap_score


def evaluate_model(model, X_test, y_test, name_prefix='IsolationForest'):
    """Evaluate the model on test data and save plots."""
    anomaly_scores, predictions = predict_scores(model, X_test)
    metrics = {
        'Model': name_prefix,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, anomaly_scores),
        'AveragePrecision': average_precision_score(y_test, anomaly_scores)
    }

    metrics['ROC_AUC'], metrics['AveragePrecision'] = plot_roc_pr_curves(y_test, anomaly_scores, name_prefix)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name_prefix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'../models/{name_prefix}_confusion_matrix.png')
    plt.close()

    return metrics


def save_model(model, filepath):
    """Save the trained model."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    data_dir = '../data/'
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    contamination = 0.1
    model = train_isolation_forest(X_train, contamination)
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)
    save_model(model, '../models/isolation_forest.pkl')