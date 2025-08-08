# PMGSY Scheme Detector

An intelligent classification system for Pradhan Mantri Gram Sadak Yojana (PMGSY) schemes using machine learning. This project automates the classification of PMGSY projects into different schemes based on their physical and financial data.

## 📋 Project Overview

PMGSY is a major rural development program with various schemes like PMGSY-I, PMGSY-II, etc. This project provides an automated solution to classify thousands of projects under these schemes, which is typically a slow and error-prone manual process.

## 🚀 Features

- Automated classification of PMGSY projects
- Web-based interface for easy interaction
- REST API for integration with other systems
- Model training and evaluation pipeline
- Data preprocessing and feature engineering
- Sample dataset and pre-trained model for quick start

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: scikit-learn, XGBoost, pandas, NumPy
- **Data Processing**: pandas, NumPy, scikit-learn
- **Model Persistence**: joblib
- **Development Tools**: Jupyter Notebook, VS Code

## 📁 Project Structure

```
PMGSY-SCHEME-Detector/
├── data/                   # Dataset files and uploads
│   └── sample_data.csv     # Sample dataset
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── static/             # Static files (CSS, JS, images)
│   │   ├── css/            # Stylesheets
│   │   └── js/             # JavaScript files
│   ├── templates/          # HTML templates
│   ├── app.py              # Main application
│   └── train_model.py      # Model training script
├── tests/                  # Test files
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── run.py                 # Application launcher
└── setup.bat              # Windows setup script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

### Installation (Windows)

1. **Clone the repository** (or download as ZIP and extract):
   ```
   git clone https://github.com/Harish2004-sonwale/PMGSY--SCHEME--Detector.git
   cd PMGSY--SCHEME--Detector
   ```

2. **Run the setup script**:
   Double-click on `setup.bat` or run it from the command prompt:
   ```
   .\setup.bat
   ```

   This will:
   - Create a Python virtual environment
   - Activate the environment
   - Install all required dependencies

### Manual Installation (Alternative)

If the setup script doesn't work, follow these steps:

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the application**:
   ```
   python run.py
   ```

2. **Access the web interface**:
   Open your web browser and go to:
   ```
   http://localhost:5000
   ```

3. **Using the application**:
   - Enter project details in the form
   - Click "Predict Scheme" to get the classification
   - For batch processing, use the "Batch Prediction" tab

## 🤖 Model Training

### Using the Sample Model

The application comes with a pre-trained sample model. If you want to train your own model:

1. Prepare your dataset in CSV format and place it in the `data/` directory

2. Train a new model:
   ```
   python -m src.train_model --data data/your_dataset.csv --model-type xgb
   ```

   Available model types: `xgb` (XGBoost) or `rf` (Random Forest)

3. The trained model will be saved in the `models/` directory

### Training with Jupyter Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `notebooks/pmgsy_scheme_classification.ipynb`

3. Follow the notebook instructions to explore the data and train models

## 🌐 API Endpoints

The application provides the following REST API endpoints:

- `POST /predict` - Classify a single project
  ```json
  {
    "projectName": "Sample Project",
    "financialData": 250.75,
    "physicalProgress": 85.5
  }
  ```

- `POST /batch_predict` - Upload a CSV file for batch prediction
  ```
  curl -X POST -F "file=@data/sample_data.csv" http://localhost:5000/batch_predict
  ```

## 📊 Sample Dataset

A sample dataset is provided in `data/sample_data.csv` with the following columns:

- `project_id`: Unique identifier for the project
- `project_name`: Name of the project
- `financial_data`: Financial allocation (in lakhs)
- `physical_progress`: Physical progress percentage
- `state`: State where the project is located
- `district`: District of the project
- `block`: Block name
- `habitation`: Habitation name
- `road_length`: Length of the road (in km)
- `scheme`: Target variable (PMGSY-I, PMGSY-II, PMGSY-III)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Special thanks to IBM SkillsBuild for the learning resources

## 🤝 Contributing

Contributions are welcome! IBM CLOUD 

