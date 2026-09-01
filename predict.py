import os
import joblib
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any

from utils.feature_engineering import add_engineered_features

MODEL_PATH = os.path.join("models", "best_churn_model.joblib")

def load_prediction_pipeline(model_path: str = MODEL_PATH):
    """
    Loads saved model artifact containing preprocessor and classifier.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run `python train.py` first.")
    artifact = joblib.load(model_path)
    return artifact

def predict_single_customer(customer_data: Dict[str, Any], pipeline_artifact: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Runs end-to-end churn prediction on a single customer dictionary.
    """
    if pipeline_artifact is None:
        pipeline_artifact = load_prediction_pipeline()
        
    preprocessor = pipeline_artifact["preprocessor"]
    model = pipeline_artifact["model"]
    num_cols = pipeline_artifact.get("num_cols", [])
    cat_cols = pipeline_artifact.get("cat_cols", [])
    
    # Convert input dict to pandas DataFrame (1 row)
    df_raw = pd.DataFrame([customer_data])
    
    # Apply feature engineering
    df_feat = add_engineered_features(df_raw)
    
    # Ensure exact dtype matching with preprocessor
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(str)
    for col in num_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0).astype(float)
    
    # Transform using preprocessor
    X_proc = preprocessor.transform(df_feat)
    
    # Predict probabilities
    proba = float(model.predict_proba(X_proc)[0, 1])
    pred_class = int(proba >= 0.5)
    
    # Assign risk level category
    if proba >= 0.70:
        risk_level = "High Risk"
        recommendation = "Offer 1-year loyalty contract discount and dedicated customer success call."
    elif proba >= 0.40:
        risk_level = "Medium Risk"
        recommendation = "Send targeted email campaign featuring upgraded service add-ons."
    else:
        risk_level = "Low Risk"
        recommendation = "Customer account stable. Maintain standard automated newsletter touchpoints."
        
    return {
        "customer_id": customer_data.get("customer_id", "UNKNOWN"),
        "predicted_churn": pred_class,
        "churn_label": "Yes" if pred_class == 1 else "No",
        "churn_probability": round(proba, 4),
        "churn_probability_percent": round(proba * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "model_used": pipeline_artifact.get("model_name", "XGBoost")
    }

def run_sample_predictions():
    """
    Runs predictions on representative High Risk and Low Risk sample customer profiles.
    """
    pipeline = load_prediction_pipeline()
    
    high_risk_customer = {
        "customer_id": "CUST-HIGH-RISK",
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
        "monthly_charges": 95.50,
        "total_charges": 191.00
    }
    
    low_risk_customer = {
        "customer_id": "CUST-LOW-RISK",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "tenure": 48,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 64.20,
        "total_charges": 3081.60
    }
    
    print("=" * 60)
    print("RUNNING DEMO PREDICTIONS")
    print("=" * 60)
    
    res1 = predict_single_customer(high_risk_customer, pipeline)
    print(f"\nProfile 1 ({high_risk_customer['customer_id']}):")
    print(json.dumps(res1, indent=2))
    
    res2 = predict_single_customer(low_risk_customer, pipeline)
    print(f"\nProfile 2 ({low_risk_customer['customer_id']}):")
    print(json.dumps(res2, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Customer Churn")
    parser.add_argument("--json", type=str, help="JSON string of customer attributes")
    parser.add_argument("--sample", action="store_true", help="Run sample prediction test")
    
    args = parser.parse_args()
    
    if args.json:
        cust_dict = json.loads(args.json)
        res = predict_single_customer(cust_dict)
        print(json.dumps(res, indent=2))
    else:
        run_sample_predictions()
