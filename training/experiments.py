import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from utils.data_preprocessing import load_data, preprocess_data, split_data, save_processed_data
from training.isolation_forest import train_isolation_forest, predict_scores as if_predict_scores, save_model as save_if_model
from training.autoencoder import train_autoencoder, get_anomaly_scores, save_model as save_ae_model


def load_processed_data():
    data_dir = os.path.join(ROOT_DIR, 'data')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true, y_pred, y_scores, name):
    """
    Evaluate predictions with metrics suitable for imbalanced data.
    Note: Accuracy is misleading in imbalanced datasets as it can be high even with poor anomaly detection.
    Focus on F1-score, Precision, Recall, ROC-AUC, and Average Precision (PR-AUC).
    """
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),  # High accuracy doesn't mean good anomaly detection
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else np.nan,
        'AveragePrecision': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else np.nan
    }
    return metrics


def plot_roc_curve(y_true, y_scores, model_name, filename):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'models', filename))
    plt.close()


def plot_pr_curve(y_true, y_scores, model_name, filename):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'models', filename))
    plt.close()


def analyze_confusion_matrix(y_true, y_pred, model_name):
    """Analyze false positives vs false negatives."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{model_name} Confusion Matrix Analysis:")
    print(f"True Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")
    print(".3f")
    print(".3f")

    # In fraud detection, FN (missed fraud) is often more costly than FP
    if fn > fp:
        print("More false negatives than false positives - model misses many anomalies.")
    else:
        print("More false positives than false negatives - model flags many normal as anomalies.")


def explain_poor_performance():
    """Explain why anomaly detection performance is poor in this dataset."""
    print("\n" + "="*60)
    print("EXPLANATION OF POOR ANOMALY DETECTION PERFORMANCE")
    print("="*60)
    print("1. SEVERE CLASS IMBALANCE:")
    print("   - Dataset has ~99% normal transactions, ~1% fraud")
    print("   - Models trained on mostly normal data struggle to learn fraud patterns")
    print("   - Random guessing would achieve ~99% accuracy, making accuracy useless")
    print()
    print("2. LIMITED FEATURES:")
    print("   - Only basic transaction features (amount, type, location)")
    print("   - No temporal patterns, user behavior, or network features")
    print("   - Fraud patterns may not be separable in this feature space")
    print()
    print("3. ANOMALY DETECTION CHALLENGES:")
    print("   - Unsupervised methods (IF, AE) don't use fraud labels during training")
    print("   - Hard to distinguish subtle fraud from normal variation")
    print("   - Threshold selection is critical but heuristic")
    print()
    print("4. EXPECTED PERFORMANCE:")
    print("   - F1-scores around 0.01-0.05 are common in real-world fraud detection")
    print("   - Focus on Recall (catching fraud) over Precision (reducing false alarms)")
    print("   - Ensemble methods and SMOTE can help but have limits")
    print("="*60)


def run_trivial_baseline(y_test):
    preds = np.zeros_like(y_test)
    scores = np.zeros_like(y_test, dtype=float)
    return evaluate_predictions(y_test, preds, scores, 'Trivial_Normal')


def run_random_baseline(y_test, seed=42):
    rng = np.random.RandomState(seed)
    scores = rng.random(size=len(y_test))
    preds = (scores > 0.5).astype(int)
    return evaluate_predictions(y_test, preds, scores, 'Random_Classifier')


def run_lof(X_train, X_test, y_test, contamination=0.01):
    lof = LocalOutlierFactor(n_neighbors=15, contamination=contamination, novelty=True)
    lof.fit(X_train)
    raw_scores = lof.decision_function(X_test)
    anomaly_scores = -raw_scores
    preds = np.where(lof.predict(X_test) == -1, 1, 0)
    return evaluate_predictions(y_test, preds, anomaly_scores, f'LOF_cont_{contamination}')


def run_if(X_train, X_test, y_test, contamination=0.01, random_state=42):
    model = train_isolation_forest(X_train, contamination=contamination, random_state=random_state)
    anomaly_scores, preds = if_predict_scores(model, X_test)
    return model, evaluate_predictions(y_test, preds, anomaly_scores, f'IF_cont_{contamination}')


def run_autoencoder(X_train, X_test, y_train, y_test, threshold_percentile=95, random_state=42):
    model = train_autoencoder(X_train, y_train, random_state=random_state)
    scores = get_anomaly_scores(model, X_test)
    threshold = np.percentile(scores, threshold_percentile)
    preds = (scores > threshold).astype(int)
    metrics = evaluate_predictions(y_test, preds, scores, f'AE_thr_{threshold_percentile}')
    metrics['Threshold'] = threshold
    return model, metrics


def run_smote_experiment(X_train, y_train, X_test, y_test, use_smote=True):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    if use_smote:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
        name = 'LogisticRegression_SMOTE'
    else:
        model.fit(X_train, y_train)
        name = 'LogisticRegression_Original'
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    return evaluate_predictions(y_test, preds, probs, name)


def aggregate_repeated_results(results):
    df = pd.DataFrame(results)
    aggregated = df.groupby('Model').agg(
        Accuracy=['mean', 'std'],
        Precision=['mean', 'std'],
        Recall=['mean', 'std'],
        F1=['mean', 'std'],
        ROC_AUC=['mean', 'std'],
        AveragePrecision=['mean', 'std']
    )
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    return aggregated.reset_index()


def sample_training_data(X_train, y_train, sample_size=10000, random_state=42):
    """Sample a representative subset from the training set for repeated experiments."""
    if len(X_train) <= sample_size:
        return X_train, y_train

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=random_state
    )
    return X_sample, y_sample


def run_hyperparameter_sensitivity(X_train, X_test, y_train, y_test, contamination_values, threshold_values, n_runs=5):
    records = []
    for contamination in contamination_values:
        for run_index in range(n_runs):
            _, metrics = run_if(X_train, X_test, y_test, contamination=contamination, random_state=42 + run_index)
            metrics['Contamination'] = contamination
            metrics['Run'] = run_index + 1
            records.append(metrics)

    df_contamination = pd.DataFrame(records)
    contamination_summary = df_contamination.groupby('Contamination').agg(
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std'),
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std'),
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std')
    ).reset_index()

    threshold_records = []
    for threshold in threshold_values:
        for run_index in range(n_runs):
            _, metrics = run_autoencoder(X_train, X_test, y_train, y_test,
                                         threshold_percentile=threshold, random_state=42 + run_index)
            metrics['ThresholdPercentile'] = threshold
            metrics['Run'] = run_index + 1
            threshold_records.append(metrics)

    df_threshold = pd.DataFrame(threshold_records)
    threshold_summary = df_threshold.groupby('ThresholdPercentile').agg(
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std'),
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std'),
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std')
    ).reset_index()

    return contamination_summary, threshold_summary


def run_ensemble(X_train, X_test, y_train, y_test, if_contamination=0.01, ae_threshold=95, random_state=42):
    if_model = train_isolation_forest(X_train, contamination=if_contamination, random_state=random_state)
    if_scores, if_preds = if_predict_scores(if_model, X_test)
    ae_model = train_autoencoder(X_train, y_train, random_state=random_state)
    ae_scores = get_anomaly_scores(ae_model, X_test)
    ae_threshold_value = np.percentile(ae_scores, ae_threshold)
    ae_preds = (ae_scores > ae_threshold_value).astype(int)

    combined_preds = np.maximum(if_preds, ae_preds)
    if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9)
    ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-9)
    combined_scores = 0.5 * if_norm + 0.5 * ae_norm

    metrics = evaluate_predictions(y_test, combined_preds, combined_scores, 'Ensemble_OR')
    metrics['IF_Contamination'] = if_contamination
    metrics['AE_ThresholdPercentile'] = ae_threshold
    return metrics


def save_table(df, filename):
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    df.to_csv(os.path.join(ROOT_DIR, 'models', filename), index=False)
    print(f"Saved results to models/{filename}")


def main():
    raw_path = os.path.join(ROOT_DIR, 'data', 'credit_card_fraud_dataset.csv')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    print('Loading and preprocessing data...')
    df = load_data(raw_path)
    X, y, scaler, le_type, le_location = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    save_processed_data(X_train, X_test, y_train, y_test, scaler, le_type, le_location,
                        os.path.join(ROOT_DIR, 'data'), os.path.join(ROOT_DIR, 'models'))

    print('Sampling a representative training subset for repeated experiments...')
    X_train_sample, y_train_sample = sample_training_data(X_train, y_train, sample_size=10000, random_state=42)

    print('Running baseline comparisons...')
    baseline_results = [
        run_trivial_baseline(y_test),
        run_random_baseline(y_test),
        run_lof(X_train, X_test, y_test, contamination=0.01),
    ]
    baseline_df = pd.DataFrame(baseline_results)
    save_table(baseline_df, 'baseline_comparison.csv')

    print('Training and evaluating Isolation Forest and Autoencoder on sample training data...')
    if_models = []
    if_results = []
    for run_index in range(5):
        model, metrics = run_if(X_train_sample, X_test, y_test, contamination=0.01, random_state=42 + run_index)
        if_models.append(model)
        metrics['Run'] = run_index + 1
        if_results.append(metrics)
    if_results_df = pd.DataFrame(if_results)
    save_table(if_results_df, 'isolation_forest_results.csv')

    ae_models = []
    ae_results = []
    for run_index in range(5):
        model, metrics = run_autoencoder(X_train_sample, X_test, y_train_sample, y_test,
                                         threshold_percentile=95, random_state=42 + run_index)
        ae_models.append(model)
        metrics['Run'] = run_index + 1
        ae_results.append(metrics)
    ae_df = pd.DataFrame(ae_results)
    save_table(ae_df, 'autoencoder_results.csv')

    print('Running hyperparameter sensitivity analysis on sampled training data...')
    contamination_summary, threshold_summary = run_hyperparameter_sensitivity(
        X_train_sample, X_test, y_train_sample, y_test,
        contamination_values=[0.001, 0.01, 0.05],
        threshold_values=[90, 95, 99],
        n_runs=5
    )
    save_table(contamination_summary, 'contamination_sensitivity.csv')
    save_table(threshold_summary, 'threshold_sensitivity.csv')

    # Explain threshold trade-offs
    print("\nTHRESHOLD TRADE-OFF ANALYSIS:")
    print("Higher thresholds (e.g., 99th percentile) → Higher Precision, Lower Recall")
    print("Lower thresholds (e.g., 90th percentile) → Higher Recall, Lower Precision")
    print("In fraud detection, prioritize Recall to catch more fraud, accepting more false positives.")
    print(f"Threshold sensitivity results saved to models/threshold_sensitivity.csv")

    print('Running ensemble evaluation on full training data...')
    ensemble_metrics = run_ensemble(X_train, X_test, y_train, y_test,
                                   if_contamination=0.01, ae_threshold=95, random_state=42)
    ensemble_df = pd.DataFrame([ensemble_metrics])
    save_table(ensemble_df, 'ensemble_results.csv')

    print('Running SMOTE experiment...')
    smote_original = run_smote_experiment(X_train, y_train, X_test, y_test, use_smote=False)
    smote_augmented = run_smote_experiment(X_train, y_train, X_test, y_test, use_smote=True)
    smote_df = pd.DataFrame([smote_original, smote_augmented])
    save_table(smote_df, 'smote_comparison.csv')

    print('Saving final full-data models into the models directory...')
    final_if_model, if_metrics = run_if(X_train, X_test, y_test, contamination=0.01, random_state=999)
    save_if_model(final_if_model, os.path.join(ROOT_DIR, 'models', 'isolation_forest.pkl'))

    final_ae_model, ae_metrics = run_autoencoder(X_train, X_test, y_train, y_test,
                                       threshold_percentile=95, random_state=999)
    save_ae_model(final_ae_model, os.path.join(ROOT_DIR, 'models', 'autoencoder.pkl'))

    # Generate ROC and PR curves for final models
    print('Generating evaluation plots...')
    if_scores, if_preds = if_predict_scores(final_if_model, X_test)
    plot_roc_curve(y_test, if_scores, 'Isolation Forest', 'isolation_forest_roc.png')
    plot_pr_curve(y_test, if_scores, 'Isolation Forest', 'isolation_forest_pr.png')

    ae_scores = get_anomaly_scores(final_ae_model, X_test)
    plot_roc_curve(y_test, ae_scores, 'Autoencoder', 'autoencoder_roc.png')
    plot_pr_curve(y_test, ae_scores, 'Autoencoder', 'autoencoder_pr.png')

    # Analyze confusion matrices
    analyze_confusion_matrix(y_test, if_preds, 'Isolation Forest')
    ae_threshold = np.percentile(ae_scores, 95)
    ae_preds = (ae_scores > ae_threshold).astype(int)
    analyze_confusion_matrix(y_test, ae_preds, 'Autoencoder')

    # Explain poor performance
    explain_poor_performance()

    print('Models and experiment results are saved in the models directory.')


if __name__ == '__main__':
    main()
