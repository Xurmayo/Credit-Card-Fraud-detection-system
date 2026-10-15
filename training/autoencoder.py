import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def build_autoencoder(input_dim, random_state=42):
    """Build Autoencoder using scikit-learn MLPRegressor."""
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16, 32, 64),
        activation='relu',
        solver='adam',
        max_iter=100,
        tol=1e-4,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False
    )
    return autoencoder


def train_autoencoder(X_train, y_train, random_state=42):
    """Train Autoencoder on normal data only."""
    normal_indices = np.where(y_train == 0)[0]
    X_normal = X_train[normal_indices]

    autoencoder = build_autoencoder(X_train.shape[1], random_state=random_state)
    print(f"Training autoencoder on {len(X_normal)} normal examples...")
    autoencoder.fit(X_normal, X_normal)
    return autoencoder


def get_anomaly_scores(model, X):
    """Compute reconstruction error for each sample."""
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    return mse


def predict_scores(model, X, threshold_percentile=95):
    """Return anomaly scores and binary labels for a threshold percentile."""
    scores = get_anomaly_scores(model, X)
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores > threshold).astype(int)
    return scores, predictions, threshold


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


def evaluate_autoencoder(model, X_test, y_test, threshold_percentile=95, name_prefix='Autoencoder'):
    """Evaluate the autoencoder using detection metrics."""
    scores, predictions, threshold = predict_scores(model, X_test, threshold_percentile)
    metrics = {
        'Model': name_prefix,
        'ThresholdPercentile': threshold_percentile,
        'ThresholdValue': threshold,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, scores),
        'AveragePrecision': average_precision_score(y_test, scores)
    }

    metrics['ROC_AUC'], metrics['AveragePrecision'] = plot_roc_pr_curves(y_test, scores, name_prefix)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name_prefix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'../models/{name_prefix}_confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, label='Scores')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold {threshold_percentile}th')
    plt.title(f'Anomaly Score Distribution - {name_prefix}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'../models/{name_prefix}_scores.png')
    plt.close()

    return metrics


def save_model(model, filepath):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    data_dir = '../data/'
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    autoencoder = train_autoencoder(X_train, y_train)
    metrics = evaluate_autoencoder(autoencoder, X_test, y_test)
    print(metrics)
    save_model(autoencoder, '../models/autoencoder.pkl')