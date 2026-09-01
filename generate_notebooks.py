import json
import os

os.makedirs("notebooks", exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

# Notebook 1: EDA
nb1_cells = [
    md_cell("# 01 - Exploratory Data Analysis (EDA)\n\n### Objective:\nPerform comprehensive EDA on the Telco Customer Churn dataset to uncover feature distributions, correlations, missing value patterns, and target class imbalance."),
    code_cell("""import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_raw_dataset, get_dataset_summary

# Set visualization style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)"""),
    md_cell("## 1. Load Dataset & Summary Statistics"),
    code_cell("""df = load_raw_dataset("../Telco_customer_churn.xlsx")
summary = get_dataset_summary(df)
print("Shape:", df.shape)
print("Churn Rate:", summary["churn_rate_percent"], "%")
df.head()"""),
    md_cell("## 2. Target Variable Class Balance"),
    code_cell("""plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="churn", palette=["#10b981", "#ef4444"])
plt.title("Target Distribution: Churn vs Retained")
plt.xticks([0, 1], ["Retained (0)", "Churned (1)"])
plt.show()"""),
    md_cell("## 3. Tenure vs Churn"),
    code_cell("""plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x="tenure", hue="churn", common_norm=False, fill=True, palette=["#10b981", "#ef4444"])
plt.title("Tenure Distribution by Churn Status")
plt.xlabel("Tenure (Months)")
plt.show()"""),
    md_cell("## 4. Contract Type Impact"),
    code_cell("""plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="contract", hue="churn", palette=["#10b981", "#ef4444"])
plt.title("Churn Rate by Contract Type")
plt.show()""")
]

with open("notebooks/01_exploratory_data_analysis.ipynb", "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# Notebook 2: Preprocessing & Feature Engineering
nb2_cells = [
    md_cell("# 02 - Preprocessing & Feature Engineering\n\n### Objective:\nHandle missing values, encode categorical features, scale numerical columns, engineer domain features, and compare metrics."),
    code_cell("""import sys
sys.path.append("..")

import pandas as pd
from utils.data_loader import load_raw_dataset
from utils.feature_engineering import add_engineered_features
from utils.preprocessing import split_and_preprocess_data"""),
    md_cell("## 1. Feature Engineering"),
    code_cell("""raw_df = load_raw_dataset("../Telco_customer_churn.xlsx")
df_engineered = add_engineered_features(raw_df)
print("New Features Created:")
print(df_engineered[["avg_monthly_charge_ratio", "total_services", "is_alone", "is_high_risk_profile"]].head())"""),
    md_cell("## 2. Preprocessing & Stratified Train/Test Split"),
    code_cell("""preproc_data = split_and_preprocess_data(df_engineered)
print("X_train processed shape:", preproc_data["X_train_proc"].shape)
print("Feature count after One-Hot Encoding:", len(preproc_data["feature_names"]))""")
]

with open("notebooks/02_preprocessing_and_feature_engineering.ipynb", "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# Notebook 3: Model Development & Tuning
nb3_cells = [
    md_cell("# 03 - Model Development & Hyperparameter Tuning\n\n### Objective:\nTrain Logistic Regression, Random Forest, and XGBoost classifiers. Perform randomized search tuning and evaluate metrics."),
    code_cell("""import sys
import os
import importlib
sys.path.append("..")

import utils.data_loader
import utils.preprocessing
import train

importlib.reload(utils.data_loader)
importlib.reload(utils.preprocessing)
importlib.reload(train)

import json
from train import train_and_evaluate_all"""),
    code_cell("""# Run complete training pipeline
metrics = train_and_evaluate_all()
print(json.dumps(metrics, indent=2))""")
]

with open("notebooks/03_model_development_and_tuning.ipynb", "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# Notebook 4: Model Explainability (SHAP)
nb4_cells = [
    md_cell("# 04 - Model Explainability using SHAP\n\n### Objective:\nExplain model predictions globally (feature importances) and locally (individual customer waterfall plots) using SHAP values."),
    code_cell("""import sys
import os
import importlib
sys.path.append("..")

import utils.data_loader
import utils.preprocessing

importlib.reload(utils.data_loader)
importlib.reload(utils.preprocessing)

import joblib
import shap
import pandas as pd
from utils.data_loader import load_raw_dataset
from utils.feature_engineering import add_engineered_features"""),
    code_cell("""# Load model artifact
artifact = joblib.load("../models/best_churn_model.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
feature_names = artifact["feature_names"]
num_cols = artifact.get("num_cols", [])
cat_cols = artifact.get("cat_cols", [])

raw_df = load_raw_dataset("../Telco_customer_churn.xlsx")
df_engineered = add_engineered_features(raw_df)

X = df_engineered.drop(columns=["customer_id", "churn"]).copy()

# Ensure exact dtype matching with preprocessor
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
for col in num_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

X_proc = preprocessor.transform(X)

# Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_proc[:500])

print("SHAP values computed for top 500 samples.")
shap.summary_plot(shap_values, X_proc[:500], feature_names=feature_names)""")
]

with open("notebooks/04_model_explainability_shap.ipynb", "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print("Jupyter notebooks generated successfully in notebooks/")
