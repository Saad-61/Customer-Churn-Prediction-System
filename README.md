# 🛡️ Customer Churn Prediction System

An end-to-end Machine Learning web application and REST API designed to predict customer churn probability, identify high-risk accounts, and provide actionable retention recommendations using **XGBoost**, **FastAPI**, **Bootstrap 5**, and **SHAP explainability**.

---

## 📌 Project Overview & Features

- **Data Processing & Feature Engineering**: Standardizes raw Telco Excel dataset (`Telco_customer_churn.xlsx`), imputes missing values, applies One-Hot Encoding and `StandardScaler`, and engineers domain ratio features (`avg_monthly_charge_ratio`, `total_services`, `is_alone`, `is_high_risk_profile`).
- **Model Development & Benchmarking**: Compares **Logistic Regression**, **Random Forest**, and **XGBoost Classifier** across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
- **Hyperparameter Optimization**: Tunes XGBoost hyperparameters using Stratified 5-Fold Cross-Validation (`RandomizedSearchCV`).
- **Model Explainability**: Leverages **SHAP (SHapley Additive exPlanations)** for global feature importances and local customer prediction explanations.
- **FastAPI REST API**: Serves prediction endpoints (`POST /predict`), health check (`GET /health`), and audit history (`GET /api/history`).
- **Interactive Web Interface**: Built with Bootstrap 5, featuring a dynamic SVG churn probability gauge meter, risk badges (`High Risk`, `Medium Risk`, `Low Risk`), and sample profile pre-fill buttons.
- **SQLite Database Audit Logger**: Automatically logs every prediction request to `predictions.db` for compliance and audit reporting.
- **Docker Containerization**: Includes `Dockerfile` and `docker-compose.yml` for single-command production deployment.
- **Unit Test Suite**: 8 automated `pytest` test cases covering data loading, feature engineering, model inference, and API endpoints.

---

## 📁 Repository Folder Structure

```
Customer-Churn-Prediction/
├── data/                       # Raw & processed data outputs
│   └── training_metrics.json   # Model benchmark metrics JSON
├── notebooks/                  # Step-by-step Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_development_and_tuning.ipynb
│   └── 04_model_explainability_shap.ipynb
├── models/                     # Serialized model artifacts
│   └── best_churn_model.joblib # Sklearn Pipeline + XGBoost Model
├── api/                        # FastAPI REST API Backend
│   ├── __init__.py
│   ├── app.py                  # API endpoints & route handlers
│   └── schemas.py              # Pydantic input/output schemas
├── templates/                  # Frontend HTML templates
│   └── index.html              # Bootstrap 5 Dashboard
├── static/                     # Web static assets
│   ├── css/style.css           # Glassmorphism dark theme CSS
│   └── js/app.js               # Dynamic gauge & AJAX logic
├── utils/                      # Modular python pipelines
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & cleaning
│   ├── preprocessing.py       # OneHot & StandardScaler pipeline
│   ├── feature_engineering.py  # Domain ratio feature creation
│   └── db_logger.py            # SQLite database logger
├── tests/                      # Pytest unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── Telco_customer_churn.xlsx   # Source dataset
├── requirements.txt            # Python dependencies
├── train.py                    # Automated end-to-end model training script
├── predict.py                  # CLI inference script
├── Dockerfile                  # Container build config
├── docker-compose.yml          # Container orchestration
└── README.md                   # System documentation
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Installation

Ensure Python 3.11+ is installed. Clone or open the project folder and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Model Training

Execute `train.py` to preprocess data, train classifiers, run hyperparameter tuning, evaluate metrics, and save model artifacts:

```bash
python train.py
```

### 3. Run CLI Sample Predictions

Test single customer churn prediction via command-line:

```bash
python predict.py --sample
```

### 4. Launch Web App & REST API Server

Start the FastAPI web server on `http://localhost:8000`:

```bash
uvicorn api.app:app --reload
```

Open your browser and navigate to:
- **Interactive Web Interface**: `http://localhost:8000`
- **Swagger Interactive API Documentation**: `http://localhost:8000/docs`
- **ReDoc API Documentation**: `http://localhost:8000/redoc`

---

## 📊 Model Evaluation Results

* 🏆 **1. Tuned XGBoost (Selected Winner)**
  * **ROC-AUC**: `0.8568` *(Highest overall risk ranking capability across all probability thresholds)*
  * **Accuracy**: `81.12%` *(Correctly classified 1,143 out of 1,409 test customers)*
  * **Precision**: `70.73%` *(Highest precision — when predicting churn, 70.7% were actual churners)*
  * **Recall**: `49.47%` *(Identified 185 out of 374 churners)*
  * **F1-Score**: `0.5818`

* 📈 **2. Logistic Regression (Baseline)**
  * **ROC-AUC**: `0.8555` | **Accuracy**: `80.84%` | **Precision**: `66.07%` | **Recall**: `57.22%` | **F1-Score**: `0.6132`

* 🚀 **3. Baseline XGBoost**
  * **ROC-AUC**: `0.8350` | **Accuracy**: `80.06%` | **Precision**: `64.33%` | **Recall**: `55.88%` | **F1-Score**: `0.5980`

* 🌲 **4. Random Forest Classifier**
  * **ROC-AUC**: `0.8327` | **Accuracy**: `78.92%` | **Precision**: `62.62%` | **Recall**: `51.07%` | **F1-Score**: `0.5626`

---

## 🔌 API Endpoint Documentation

### `POST /predict`
Submits customer profile attributes and returns predicted churn score, risk level, and recommendations.

**Request Payload Example:**
```json
{
  "customer_id": "CUST-9921",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No",
  "tenure": 2,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 94.85,
  "total_charges": 189.70
}
```

**Response Example:**
```json
{
  "status": "success",
  "customer_id": "CUST-9921",
  "predicted_churn": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6986,
  "churn_probability_percent": 69.86,
  "risk_level": "Medium Risk",
  "recommendation": "Send targeted email campaign featuring upgraded service add-ons.",
  "model_used": "Tuned XGBoost",
  "logged_db_id": 1
}
```

### `GET /health`
Returns system status and loaded model details.

### `GET /api/history`
Returns recent SQLite database prediction logs.

---

## 🧪 Running Automated Unit Tests

Run the pytest test suite to verify code quality:

```bash
pytest tests/
```

---

## 🐳 Docker Deployment

To build and run the system inside a Docker container:

```bash
docker-compose up --build
```

Access the application at `http://localhost:8000`.
