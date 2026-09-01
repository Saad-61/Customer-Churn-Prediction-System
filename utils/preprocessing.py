import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any

CATEGORICAL_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method"
]

NUMERICAL_COLUMNS = [
    "tenure", "monthly_charges", "total_charges"
]

# Additional columns if engineered features exist
ENGINEERED_NUMERICAL = [
    "avg_monthly_charge_ratio", "charge_difference", "tenure_years", "total_services"
]
ENGINEERED_CATEGORICAL = [
    "is_new_customer", "is_loyal_customer", "is_alone", "is_high_risk_profile"
]

def prepare_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies available numerical and categorical columns present in dataframe.
    """
    all_num = NUMERICAL_COLUMNS + ENGINEERED_NUMERICAL
    all_cat = CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL
    
    num_cols = [c for c in all_num if c in df.columns]
    cat_cols = [c for c in all_cat if c in df.columns]
    
    return num_cols, cat_cols

def build_preprocessor_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline.
    
    Concepts:
    1. StandardScaler: Centers numerical values (\mu=0) and scales variance (\sigma=1).
    2. OneHotEncoder: Converts categorical strings into binary indicator columns (0 or 1).
    """
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_feature_names_out(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """
    Extracts column names output by ColumnTransformer after One-Hot Encoding.
    """
    feature_names = list(num_cols)
    if "cat" in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        feature_names.extend(encoded_cat_names)
    return feature_names

def split_and_preprocess_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs Stratified Train/Test split and fits preprocessor pipeline.
    Prevents data leakage by fitting transformers strictly on X_train.
    """
    drop_cols = ["customer_id", "churn"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df["churn"].values
    
    num_cols, cat_cols = prepare_feature_lists(X)
    
    # Cast categorical columns to str and numerical columns to float for strict type consistency
    for col in cat_cols:
        X[col] = X[col].astype(str)
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
    
    # Stratified split ensures exact class distribution in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    
    # Fit preprocessor on training data ONLY to avoid data leakage
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = get_feature_names_out(preprocessor, num_cols, cat_cols)
    
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }
