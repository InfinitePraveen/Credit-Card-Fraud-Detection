import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(filepath="../data/creditcard.csv"):
    """
    Load credit card fraud detection dataset.

    Args:
        filepath (str): Path to the dataset

    Returns:
        tuple: (X, y) features and target
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please download from Kaggle."
        )

    df = pd.read_csv(filepath)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data: scale features and split into train/test.

    Args:
        X (DataFrame): Features
        y (Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Scale Amount and Time features
    X_scaled = X.copy()
    scaler = StandardScaler()
    X_scaled[["Amount", "Time"]] = scaler.fit_transform(X_scaled[["Amount", "Time"]])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler, directory="../data/"):
    """
    Save processed data to disk.

    Args:
        X_train, X_test, y_train, y_test: Data splits
        scaler: Fitted StandardScaler
        directory (str): Directory to save files
    """
    os.makedirs(directory, exist_ok=True)

    joblib.dump(X_train, os.path.join(directory, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(directory, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(directory, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(directory, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(directory, "scaler.pkl"))

    print(f"Data saved to {directory}")


if __name__ == "__main__":
    # Example usage
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Fraud in train: {y_train.sum()}")
    print(f"Fraud in test: {y_test.sum()}")
