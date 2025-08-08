#!/usr/bin/env python3
"""
Script to train a sample model using the sample dataset.
This is for demonstration purposes only.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create models directory if it doesn't exist
os.makedirs(os.path.join('models'), exist_ok=True)

def generate_sample_model():
    """Generate a sample model using the sample dataset."""
    print("Generating sample model...")
    
    # Load sample data
    data_path = os.path.join('data', 'sample_data.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df[['financial_data', 'physical_progress']]
    y = df['scheme']
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    # Save the model and artifacts
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': X.columns.tolist()
    }
    
    model_path = os.path.join('models', 'pmgsy_model.pkl')
    joblib.dump(artifacts, model_path)
    
    print(f"Sample model saved to {model_path}")
    print("Model training complete!")

if __name__ == "__main__":
    generate_sample_model()
