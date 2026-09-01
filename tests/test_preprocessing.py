import pytest
import pandas as pd
import numpy as np
from utils.data_loader import load_raw_dataset, get_dataset_summary
from utils.feature_engineering import add_engineered_features
from utils.preprocessing import split_and_preprocess_data

def test_data_loader():
    df = load_raw_dataset("Telco_customer_churn.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()
    
    summary = get_dataset_summary(df)
    assert summary["total_customers"] > 0
    assert summary["churn_rate_percent"] >= 0

def test_feature_engineering():
    raw_df = load_raw_dataset("Telco_customer_churn.xlsx")
    df_feat = add_engineered_features(raw_df)
    
    assert "avg_monthly_charge_ratio" in df_feat.columns
    assert "total_services" in df_feat.columns
    assert "is_alone" in df_feat.columns
    assert "is_high_risk_profile" in df_feat.columns
    assert not df_feat["total_services"].isnull().any()

def test_preprocessing_pipeline():
    raw_df = load_raw_dataset("Telco_customer_churn.xlsx")
    df_feat = add_engineered_features(raw_df)
    
    result = split_and_preprocess_data(df_feat, test_size=0.2, random_state=42)
    
    X_train = result["X_train_proc"]
    X_test = result["X_test_proc"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    
    assert len(X_train) + len(X_test) == len(df_feat)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert len(result["feature_names"]) == X_train.shape[1]
