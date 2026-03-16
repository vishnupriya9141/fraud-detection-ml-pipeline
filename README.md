# Credit Card Fraud Detection ML Pipeline

A cloud-based data science and machine learning pipeline for detecting fraudulent credit card transactions. This project integrates several stages, including data ingestion, preprocessing, exploratory data analysis, machine learning model development, and a REST API for real-time predictions.

## Table of Contents

- [Introduction](#introduction)
- [Business Context](#business-context)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Pipeline Overview](#pipeline-overview)
- [Dataset](#dataset)
- [Monitoring & Logging](#monitoring--logging)
- [License](#license)

---

## Introduction

The rapid expansion of digital payment systems, online banking, and e-commerce platforms has significantly increased the number of financial transactions conducted electronically. While these technologies provide convenience and efficiency, they have also created opportunities for fraudulent activities.

This project focuses on developing an **automated fraud detection system** capable of identifying suspicious credit card transactions using machine learning techniques.

---

## Business Context

### Problem Statement
Credit card fraud occurs when unauthorized individuals gain access to a cardholder's details and perform illegal transactions. With millions of transactions occurring daily, financial institutions require automated systems capable of detecting fraudulent activities quickly and accurately.

### Business Objective
- Detect fraudulent transactions with high accuracy
- Reduce financial losses caused by fraud
- Improve the efficiency of fraud monitoring systems
- Assist financial institutions in enhancing transaction security

### Analytical Objective
Build a **binary classification model** that categorizes transactions into:
- **0 → Normal Transaction**
- **1 → Fraudulent Transaction**

### Key Challenges
- **Class Imbalance**: Fraudulent transactions represent only ~0.17% of total transactions
- **False Positives**: Minimizing legitimate transactions incorrectly flagged as fraudulent

---

## Project Structure

```
Assignment/
├── data/
│   └── creditcard.csv          # Credit Card Fraud Detection Dataset
├── logs/
│   ├── api.log                 # API request logs
│   ├── data_pipeline.log       # Data pipeline execution logs
│   └── pipeline.log            # General pipeline logs
├── models/
│   └── fraud_detection_model.pkl  # Trained ML model
├── notebook/
│   └── fraud-detection-ml-pipeline.ipynb  # Jupyter notebook with full ML pipeline
├── src/
│   └── app.py                  # Flask API application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Features

- **Real-time Fraud Detection API** - Make predictions via REST API
- **End-to-end ML Pipeline** - From data ingestion to model deployment
- **Automated Data Processing** - Scheduled pipeline runs (every 3 minutes)
- **Model Monitoring** - Track performance metrics
- **Comprehensive Logging** - Track API requests and pipeline execution
- **Jupyter Notebook** - Full exploratory data analysis and model development

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python |
| **Web Framework** | Flask |
| **Machine Learning** | scikit-learn, XGBoost |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Model Serialization** | joblib |
| **API Server** | uvicorn |

---

## Installation

1. **Clone the repository**
   ```bash
   cd fraud-detection-ml-pipeline

   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the API Server

Start the Flask application:

```bash
cd src
python app.py
```

The API will be available at `http://localhost:5000`

### Making Predictions

Use the `/predict` endpoint to make fraud detection predictions:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.3, ...]}'
```

**Note**: The model expects 30 numerical features (V1-V28, scaled_amount, scaled_time).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - API status |
| `/predict` | POST | Make a fraud prediction |
| `/model-info` | GET | Get model information |
| `/pipeline-info` | GET | Get pipeline information |

### Example: Get Model Information

```bash
curl http://localhost:5000/model-info
```

Response:
```json
{
  "model_type": "Fraud Detection Classifier",
  "algorithms_used": ["Random Forest", "XGBoost"],
  "dataset": "Credit Card Fraud Detection (Kaggle)",
  "features": "30 numerical features including PCA components"
}
```

---

## Model Information

- **Model Type**: Ensemble (Random Forest + XGBoost)
- **Features**: 30 numerical features (V1-V28 from PCA + scaled_amount + scaled_time)
- **Dataset**: Credit Card Fraud Detection (Kaggle)
- **Training Data**: 284,807 transactions
- **Class Distribution**: ~0.17% fraudulent, ~99.83% normal

---

## Pipeline Overview

### Data Pipeline
1. **Data Ingestion** - Load raw transaction data
2. **Data Preprocessing** - Clean and normalize data
3. **Exploratory Data Analysis** - Analyze patterns and correlations
4. **DataOps Automation** - Scheduled pipeline runs (every 3 minutes)

### ML Pipeline
1. **Model Preparation** - Feature engineering and selection
2. **Model Training** - Train ensemble model
3. **Model Evaluation** - Evaluate using multiple metrics
4. **MLOps Monitoring** - Track model performance

### Evaluation Metrics
- **Accuracy** – Overall correctness of predictions
- **Precision** – Proportion of predicted fraud cases that are actually fraudulent
- **Recall** – Ability to correctly detect fraudulent transactions
- **F1 Score** – Harmonic mean of precision and recall

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle:

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 31 columns (Time, V1-V28, Amount, Class)

### Features Description
| Feature | Description |
|---------|-------------|
| Time | Seconds elapsed between this transaction and the first transaction |
| V1-V28 | Anonymized PCA-transformed features |
| Amount | Transaction amount |
| Class | Target variable (1 = fraud, 0 = normal) |

---

## Monitoring & Logging

The application includes comprehensive logging:

- **API Logs** (`logs/api.log`) - Track API requests and predictions
- **Pipeline Logs** (`logs/pipeline.log`) - Monitor pipeline execution
- **Data Pipeline Logs** (`logs/data_pipeline.log`) - Data processing logs

---

## License

This project is for educational and demonstration purposes.

---

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/)
- Inspired by machine learning best practices for fraud detection
