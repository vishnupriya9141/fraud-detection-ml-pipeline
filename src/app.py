# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from pathlib import Path


# ---------------------------------------------------------
# Define project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "api.log"


# ---------------------------------------------------------
# Configure Logging
# ---------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("API service started")


# ---------------------------------------------------------
# Initialize Flask Application
# ---------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------
# Load trained ML model
# ---------------------------------------------------------
try:

    print("Loading model from:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found")

    model = joblib.load(MODEL_PATH)

    logging.info("Model loaded successfully")

except Exception as e:

    logging.error(f"Error loading model: {e}")
    print("Model loading error:", e)

    model = None


# ---------------------------------------------------------
# Home Route
# ---------------------------------------------------------
@app.route('/')
def home():
    return "Credit Card Fraud Detection API is running"


# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # Convert features into numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Model prediction
        prediction = model.predict(features)

        result = int(prediction[0])

        if result == 1:
            output = "Fraudulent Transaction"
        else:
            output = "Normal Transaction"

        logging.info(f"Prediction made: {output}")

        return jsonify({
            "prediction": output
        })

    except Exception as e:

        logging.error(f"Prediction error: {str(e)}")

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# API 1: Model Information
# ---------------------------------------------------------
@app.route('/model-info', methods=['GET'])
def model_info():

    info = {
        "model_type": "Fraud Detection Classifier",
        "algorithms_used": ["Random Forest", "XGBoost"],
        "dataset": "Credit Card Fraud Detection (Kaggle)",
        "features": "30 numerical features including PCA components"
    }

    return jsonify(info)


# ---------------------------------------------------------
# API 2: Pipeline Information
# ---------------------------------------------------------
@app.route('/pipeline-info', methods=['GET'])
def pipeline_info():

    info = {
        "data_pipeline": [
            "Data Ingestion",
            "Data Preprocessing",
            "Exploratory Data Analysis",
            "DataOps Automation (every 3 minutes)"
        ],
        "ml_pipeline": [
            "Model Preparation",
            "Model Training",
            "Model Evaluation",
            "MLOps Monitoring"
        ],
        "monitoring_metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    }

    return jsonify(info)


# ---------------------------------------------------------
# Run Flask Application
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)