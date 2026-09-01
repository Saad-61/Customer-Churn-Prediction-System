import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from utils.data_loader import load_raw_dataset, get_dataset_summary
from utils.feature_engineering import add_engineered_features
from utils.preprocessing import split_and_preprocess_data

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict[str, Any]:
    """
    Evaluates classification metrics for a trained model.
    
    Concepts:
    - Accuracy: Total % of correct predictions.
    - Precision: Out of predicted churners, how many actually churned?
    - Recall (Sensitivity): Out of actual churners, how many did we catch? (Most critical for churn prevention!)
    - F1-Score: Harmonic mean of Precision and Recall.
    - ROC-AUC: Probability that model ranks a random positive instance higher than a random negative instance.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm
    }

def train_and_evaluate_all():
    print("=" * 60)
    print("STEP 1: DATA COLLECTION & EDA")
    print("=" * 60)
    raw_df = load_raw_dataset("Telco_customer_churn.xlsx")
    summary = get_dataset_summary(raw_df)
    print(f"Dataset shape: {raw_df.shape}")
    print(f"Total customers: {summary['total_customers']}")
    print(f"Churned customers: {summary['churned_customers']} ({summary['churn_rate_percent']}%)")
    
    # Ensure data export directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 2 & 3: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    # Add domain engineered features
    df_engineered = add_engineered_features(raw_df)
    print(f"Engineered dataset shape: {df_engineered.shape}")
    
    # Split and preprocess data
    preproc_result = split_and_preprocess_data(df_engineered)
    X_train = preproc_result["X_train_proc"]
    X_test = preproc_result["X_test_proc"]
    y_train = preproc_result["y_train"]
    y_test = preproc_result["y_test"]
    preprocessor = preproc_result["preprocessor"]
    feature_names = preproc_result["feature_names"]
    
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Processed feature count: {len(feature_names)}")
    
    print("\n" + "=" * 60)
    print("STEP 4: MODEL DEVELOPMENT & BASELINE EVALUATION")
    print("=" * 60)
    
    # Initialize baseline models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        print(f"  -> Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
    print("\n" + "=" * 60)
    print("STEP 5: HYPERPARAMETER TUNING (XGBoost)")
    print("=" * 60)
    print("Tuning XGBoost with Stratified 5-Fold Cross Validation...")
    
    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "scale_pos_weight": [1.0, 2.0, 2.7]  # Handle class imbalance (approx 1:2.7 ratio)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_distributions=xgb_param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_xgb = search.best_estimator_
    print(f"Best XGBoost Hyperparameters: {search.best_params_}")
    
    tuned_metrics = evaluate_model(best_xgb, X_test, y_test, "Tuned XGBoost")
    results["Tuned XGBoost"] = tuned_metrics
    fitted_models["Tuned XGBoost"] = best_xgb
    print(f"  -> Tuned Accuracy: {tuned_metrics['accuracy']:.4f} | Recall: {tuned_metrics['recall']:.4f} | F1: {tuned_metrics['f1_score']:.4f} | ROC-AUC: {tuned_metrics['roc_auc']:.4f}")
    
    # Find overall best model based on ROC-AUC & Recall score
    best_model_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_model_name]
    print(f"\n[WINNER] Top Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save combined artifact (preprocessor + model + metadata)
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "num_cols": preproc_result["num_cols"],
        "cat_cols": preproc_result["cat_cols"],
        "metrics": results[best_model_name]
    }
    
    model_path = os.path.join(models_dir, "best_churn_model.joblib")
    joblib.dump(pipeline_artifact, model_path)
    print(f"Model artifact successfully saved to: {model_path}")
    
    # Save metrics JSON for documentation & frontend
    metrics_path = os.path.join(data_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    return results

if __name__ == "__main__":
    train_and_evaluate_all()
