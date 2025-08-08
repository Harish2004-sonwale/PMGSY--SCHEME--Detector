import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model (placeholder - will be loaded on first request)
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    try:
        model_path = os.path.join('..', 'models', 'pmgsy_model.pkl')
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None


# Load model when the app starts
load_model()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from form
        data = request.get_json()
        
        # Convert to DataFrame for prediction
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            "prediction": prediction[0],
            "confidence": 0.95  # Placeholder - implement confidence calculation
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction from uploaded file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process the file
            df = pd.read_csv(filepath)
            
            # Make predictions (placeholder)
            # predictions = model.predict(df)
            # df['predicted_scheme'] = predictions
            
            # Save results
            output_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            # df.to_csv(output_path, index=False)
            
            return jsonify({
                "message": "File processed successfully",
                "download_link": f"/download/{output_filename}"
            })
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return jsonify({"error": str(e)}), 400


@app.route('/download/<filename>')
def download_file(filename):
    """Allow downloading of prediction results."""
    return redirect(url_for('static', filename=f'uploads/{filename}'))


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
