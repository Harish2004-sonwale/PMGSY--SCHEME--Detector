#!/usr/bin/env python3
"""
PMGSY Scheme Classification Model Training Script

This script trains a machine learning model to classify PMGSY projects into different schemes
based on their features. It includes data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_data(filepath):
    """
    Load and preprocess the PMGSY dataset.
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        tuple: (X, y, feature_names, label_encoder)
    """
    print(f"Loading data from {filepath}...")
    
    # Load the dataset
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Separate features and target
    # Note: Update these column names based on your actual dataset
    target_column = 'scheme'  # Replace with your target column name
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, X.columns.tolist(), label_encoder


def preprocess_data(X, y):
    """
    Preprocess the data by handling missing values, scaling features, etc.
    
    Args:
        X (DataFrame): Features
        y (array): Target variable
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\nPreprocessing data...")
    
    # Handle missing values
    print("Handling missing values...")
    X = X.fillna(X.median())
    
    # Convert categorical variables to numerical using one-hot encoding
    print("Converting categorical variables...")
    X = pd.get_dummies(X)
    
    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, model_type='xgb'):
    """
    Train a classification model.
    
    Args:
        X_train (array): Training features
        y_train (array): Training target
        model_type (str): Type of model to train ('xgb' or 'rf')
        
    Returns:
        model: Trained model
    """
    print(f"\nTraining {model_type.upper()} model...")
    
    if model_type.lower() == 'xgb':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    elif model_type.lower() == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test (array): Test features
        y_test (array): True labels
        label_encoder: Fitted label encoder
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save the figure
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")


def save_model(model, scaler, label_encoder, feature_names, model_name='pmgsy_model'):
    """
    Save the trained model and related artifacts.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        label_encoder: Fitted label encoder
        feature_names (list): List of feature names
        model_name (str): Base name for saved files
    """
    # Create a dictionary with all artifacts
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    # Save the artifacts
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    joblib.dump(artifacts, model_path)
    print(f"\nModel and artifacts saved to {model_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a PMGSY scheme classification model')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model-type', type=str, default='xgb', 
                       choices=['xgb', 'rf'], 
                       help='Type of model to train (xgb or rf)')
    parser.add_argument('--output', type=str, default='pmgsy_model',
                       help='Base name for output model files')
    
    args = parser.parse_args()
    
    # Load and preprocess the data
    X, y, feature_names, label_encoder = load_data(args.data)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train the model
    model = train_model(X_train, y_train, args.model_type)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save the model and artifacts
    save_model(model, scaler, label_encoder, feature_names, args.output)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
