import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filepath):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, encode categorical, normalize."""
    # Drop TransactionID and TransactionDate as they are not useful for modeling
    df = df.drop(['TransactionID', 'TransactionDate'], axis=1)

    # Handle missing values (if any)
    df = df.dropna()

    # Encode categorical variables
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    df['TransactionType'] = le_type.fit_transform(df['TransactionType'])
    df['Location'] = le_location.fit_transform(df['Location'])

    # Separate features and labels
    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_type, le_location

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, scaler, le_type, le_location, data_dir, models_dir):
    """Save processed data and preprocessing objects."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(le_type, os.path.join(models_dir, 'le_type.pkl'))
    joblib.dump(le_location, os.path.join(models_dir, 'le_location.pkl'))

if __name__ == "__main__":
    # Load data
    data_path = '../data/credit_card_fraud_dataset.csv'
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Preprocess
    X, y, scaler, le_type, le_location = preprocess_data(df)
    print(f"Processed features shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Save
    save_processed_data(X_train, X_test, y_train, y_test, scaler, le_type, le_location, '../data/', '../models/')
    print("Data processed and saved.")