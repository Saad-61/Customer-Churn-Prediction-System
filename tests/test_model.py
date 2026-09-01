import pytest
import os
import joblib
from predict import predict_single_customer, load_prediction_pipeline

def test_model_artifact_loading():
    model_path = os.path.join("models", "best_churn_model.joblib")
    if os.path.exists(model_path):
        pipeline = load_prediction_pipeline(model_path)
        assert "preprocessor" in pipeline
        assert "model" in pipeline
        assert "feature_names" in pipeline

def test_prediction_output_structure():
    sample_customer = {
        "customer_id": "TEST-101",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 3,
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
        "monthly_charges": 90.0,
        "total_charges": 270.0
    }
    
    model_path = os.path.join("models", "best_churn_model.joblib")
    if os.path.exists(model_path):
        res = predict_single_customer(sample_customer)
        assert res["customer_id"] == "TEST-101"
        assert res["predicted_churn"] in [0, 1]
        assert 0.0 <= res["churn_probability"] <= 1.0
        assert res["risk_level"] in ["High Risk", "Medium Risk", "Low Risk"]
