import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain feature engineering to Telco Customer Churn dataset.
    
    Concepts:
    1. Avg Monthly Charge Ratio (Total Charges / (tenure + 1)): Measures rate changes over time.
    2. Total Active Services Count: Customers with more active services have higher switching costs and lower churn.
    3. Is Alone: Single customers without partners or dependents are statistically more mobile and prone to churn.
    4. High Risk Customer Indicator: Month-to-month contract with Electronic Check payment method.
    """
    df_feat = df.copy()
    
    # 1. Average Monthly Charge Ratio
    if "total_charges" in df_feat.columns and "tenure" in df_feat.columns:
        df_feat["avg_monthly_charge_ratio"] = df_feat["total_charges"] / (df_feat["tenure"] + 1.0)
        # Difference between actual monthly charge and historic average monthly charge
        if "monthly_charges" in df_feat.columns:
            df_feat["charge_difference"] = df_feat["monthly_charges"] - df_feat["avg_monthly_charge_ratio"]
            
    # 2. Tenure Cohorts
    if "tenure" in df_feat.columns:
        df_feat["tenure_years"] = (df_feat["tenure"] / 12.0).round(2)
        df_feat["is_new_customer"] = (df_feat["tenure"] <= 6).astype(int)
        df_feat["is_loyal_customer"] = (df_feat["tenure"] >= 24).astype(int)
        
    # 3. Total Services Count
    service_cols = [
        "phone_service", "multiple_lines", "online_security", 
        "online_backup", "device_protection", "tech_support", 
        "streaming_tv", "streaming_movies"
    ]
    
    service_count = np.zeros(len(df_feat))
    for col in service_cols:
        if col in df_feat.columns:
            # Count any positive service ('Yes')
            service_count += (df_feat[col].astype(str).str.lower() == "yes").astype(int)
            
    # Add Internet Service itself if not 'No'
    if "internet_service" in df_feat.columns:
        service_count += (df_feat["internet_service"].astype(str).str.lower().isin(["dsl", "fiber optic"])).astype(int)
        
    df_feat["total_services"] = service_count
    
    # 4. Is Alone Indicator (No partner AND no dependents)
    if "partner" in df_feat.columns and "dependents" in df_feat.columns:
        no_partner = df_feat["partner"].astype(str).str.lower() == "no"
        no_dependents = df_feat["dependents"].astype(str).str.lower() == "no"
        df_feat["is_alone"] = (no_partner & no_dependents).astype(int)
        
    # 5. High Risk Profile (Month-to-month + Electronic check)
    if "contract" in df_feat.columns and "payment_method" in df_feat.columns:
        m2m = df_feat["contract"].astype(str).str.lower().str.contains("month")
        echeck = df_feat["payment_method"].astype(str).str.lower().str.contains("electronic")
        df_feat["is_high_risk_profile"] = (m2m & echeck).astype(int)
        
    return df_feat
