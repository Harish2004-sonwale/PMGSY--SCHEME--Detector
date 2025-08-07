#!/usr/bin/env python3
"""
Run the PMGSY Scheme Detector application.

This script initializes the application and starts the development server.
"""

import os
import sys
import argparse
from src.app import app

def main():
    """Run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the PMGSY Scheme Detector')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run the server on (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/uploads', exist_ok=True)
    
    # Check if model exists, if not, train a sample one
    model_path = os.path.join('models', 'pmgsy_model.pkl')
    if not os.path.exists(model_path):
        print("No trained model found. Training a sample model...")
        try:
            from scripts.train_sample_model import generate_sample_model
            generate_sample_model()
            print("Sample model trained successfully!")
        except Exception as e:
            print(f"Error training sample model: {e}")
            print("Please train a model using the train_model.py script first.")
            sys.exit(1)
    
    # Run the application
    print(f"Starting PMGSY Scheme Detector on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
