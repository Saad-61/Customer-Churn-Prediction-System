import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

COLUMN_MAPPING = {
    "CustomerID": "customer_id",
    "Customer ID": "customer_id",
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure",
    "Tenure": "tenure",
    "tenure": "tenure",
    "Phone Service": "phone_service",
    "PhoneService": "phone_service",
    "Multiple Lines": "multiple_lines",
    "MultipleLines": "multiple_lines",
    "Internet Service": "internet_service",
    "InternetService": "internet_service",
    "Online Security": "online_security",
    "OnlineSecurity": "online_security",
    "Online Backup": "online_backup",
    "OnlineBackup": "online_backup",
    "Device Protection": "device_protection",
    "DeviceProtection": "device_protection",
    "Tech Support": "tech_support",
    "TechSupport": "tech_support",
    "Streaming TV": "streaming_tv",
    "StreamingTV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "PaperlessBilling": "paperless_billing",
    "Payment Method": "payment_method",
    "PaymentMethod": "payment_method",
    "Monthly Charges": "monthly_charges",
    "MonthlyCharges": "monthly_charges",
    "Total Charges": "total_charges",
    "TotalCharges": "total_charges",
    "Churn Value": "churn",
    "Churn Label": "churn_label_str",
    "Churn": "churn"
}

def load_raw_dataset(file_path: str = "Telco_customer_churn.xlsx") -> pd.DataFrame:
    """
    Loads raw Telco Customer Churn dataset from Excel or CSV file.
    Normalizes column names to standard snake_case.
    """
    if not os.path.exists(file_path):
        # Resolve absolute project root directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path1 = os.path.join(base_dir, file_path)
        alt_path2 = os.path.join(base_dir, "data", file_path)
        alt_path3 = os.path.join("..", file_path)
        alt_path4 = os.path.join("..", "data", file_path)
        
        if os.path.exists(alt_path1):
            file_path = alt_path1
        elif os.path.exists(alt_path2):
            file_path = alt_path2
        elif os.path.exists(alt_path3):
            file_path = alt_path3
        elif os.path.exists(alt_path4):
            file_path = alt_path4
        else:
            raise FileNotFoundError(f"Dataset file not found at {file_path}. Looked in: {alt_path1}")
    
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Rename columns using mapping
    new_cols = {}
    for col in df.columns:
        col_clean = col.strip()
        if col_clean in COLUMN_MAPPING:
            new_cols[col] = COLUMN_MAPPING[col_clean]
        else:
            new_cols[col] = col_clean.lower().replace(" ", "_")
    
    df = df.rename(columns=new_cols)
    
    # Drop metadata columns that are non-features or cause data leakage
    drop_metadata = [
        "count", "country", "state", "city", "zip_code", "lat_long",
        "latitude", "longitude", "churn_label_str", "churn_score", "cltv", "churn_reason"
    ]
    df = df.drop(columns=[c for c in drop_metadata if c in df.columns])
    
    # Clean target variable 'churn'
    if "churn" in df.columns:
        if isinstance(df["churn"], pd.DataFrame):
            # Take the first churn column if duplicated
            df["churn"] = df["churn"].iloc[:, 0]
        if df["churn"].dtype == object:
            df["churn"] = df["churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)

    
    # Clean total_charges (handle empty spaces ' ')
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")
        # Impute missing total_charges with 0.0 (typically new customers with tenure = 0)
        df["total_charges"] = df["total_charges"].fillna(0.0)

    # Clean tenure and monthly_charges
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    if "monthly_charges" in df.columns:
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce").fillna(0.0)
        
    return df

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns structured data statistics for EDA analysis.
    """
    total_rows = len(df)
    churn_count = int(df["churn"].sum()) if "churn" in df.columns else 0
    churn_rate = float(churn_count / total_rows) if total_rows > 0 else 0.0
    
    missing_vals = df.isnull().sum().to_dict()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    return {
        "total_customers": total_rows,
        "churned_customers": churn_count,
        "retained_customers": total_rows - churn_count,
        "churn_rate_percent": round(churn_rate * 100, 2),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": missing_vals,
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

if __name__ == "__main__":
    df = load_raw_dataset()
    summary = get_dataset_summary(df)
    print("Dataset loaded successfully. Shape:", df.shape)
    print("Summary:", summary)
