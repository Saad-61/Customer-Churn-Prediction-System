# 📈 Customer Churn Prediction System - Technical Project Report

**Author**: Saad Asif 
**Dataset**: IBM Telco Customer Churn (`Telco_customer_churn.xlsx`)  
**Stack**: Python 3.11, Scikit-Learn, XGBoost, SHAP, FastAPI, Bootstrap 5, SQLite, Docker  

---

## 1. Executive Summary

Customer churn is one of the most critical operational challenges facing modern telecommunications companies. Acquiring a new customer typically costs **5x to 7x more** than retaining an existing customer.

This project delivers an end-to-end Machine Learning solution that predicts individual customer churn probability, categorizes risk levels (`High Risk`, `Medium Risk`, `Low Risk`), and provides actionable retention recommendations. 

The system achieves an **ROC-AUC score of 0.8568** using a tuned **XGBoost Classifier**, served via a high-performance **FastAPI REST API** and an interactive **Bootstrap 5 web dashboard**.

---

## 2. Dataset Overview & Exploratory Data Analysis (EDA)

The dataset contains **7,043 customer records** with 21 initial columns detailing customer demographics, subscribed services, account details, and churn status.

### Key EDA Findings:
1. **Target Class Distribution**:
   - Total Customers: **7,043**
   - Retained (0): **5,174 (73.46%)**
   - Churned (1): **1,869 (26.54%)**
   - *Insight*: Moderate class imbalance (1:2.77 ratio), necessitating stratified splitting and weighted loss metrics during training.
2. **Contract Type Impact**:
   - Customers on **Month-to-month contracts** exhibit the highest churn rate (~42.7%).
   - Customers on **Two-year contracts** have a near-zero churn rate (~2.8%).
3. **Tenure & Monthly Charges**:
   - New customers within their first 6 months have a significantly elevated churn probability.
   - High monthly charges (>$80/month) combined with Fiber Optic service strongly correlate with customer departure.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Data Preprocessing Pipeline
To maintain data hygiene and prevent data leakage:
- **Missing Values**: Empty spaces in `Total Charges` (for new customers with `tenure = 0`) were imputed with `0.0`.
- **Categorical Encoding**: One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`) was applied to nominal features (`gender`, `internet_service`, `contract`, `payment_method`).
- **Feature Scaling**: Numerical features (`tenure`, `monthly_charges`, `total_charges`) were normalized using `StandardScaler` (\(\mu=0, \sigma=1\)).
- **Stratified Split**: 80/20 train/test split preserving exact target proportions.

### 3.2 Feature Engineering
Synthesized domain features improved model performance:
1. **Average Monthly Charge Ratio**: \(\frac{\text{Total Charges}}{\text{tenure} + 1.0}\) — Detects historical price hikes.
2. **Total Services Count**: Sum of active add-on services (Online Security, Backup, Tech Support, Device Protection, Streaming). Higher counts increase customer switching costs.
3. **Is Alone Indicator**: Binary indicator for customers without a partner AND without dependents.
4. **High Risk Profile**: Combination of Month-to-Month contract and Electronic Check payment method.

---

## 4. Machine Learning Model Development & Tuning

Three model architectures were trained and benchmarked against standard classification metrics:

1. **Logistic Regression**: Linear baseline probabilistic model.
2. **Random Forest Classifier**: Ensemble of 100 decision trees using bagging.
3. **XGBoost Classifier**: Extreme Gradient Boosted decision trees.

### 4.1 Hyperparameter Tuning
Randomized Search Cross-Validation (`RandomizedSearchCV`) with 5-Fold Stratified K-Fold was conducted on XGBoost:
- `n_estimators`: 200
- `max_depth`: 5
- `learning_rate`: 0.01
- `subsample`: 0.9
- `colsample_bytree`: 0.9

### 4.2 Benchmark Model Performance Breakdown

* 🏆 **1. Tuned XGBoost (Selected Winner)**
  * **ROC-AUC**: `0.8568` *(Highest overall risk ranking capability across all probability thresholds)*
  * **Accuracy**: `81.12%` *(Correctly classified 1,143 out of 1,409 test customers)*
  * **Precision**: `70.73%` *(Highest precision — when predicting churn, 70.7% were actual churners with lowest false alarm rate!)*
  * **Recall**: `49.47%` *(Identified 185 out of 374 churners at default 0.50 threshold)*
  * **F1-Score**: `0.5818`
  * **Confusion Matrix**: 958 True Retained (TN) | 77 False Alarms (FP) | 189 Missed Churners (FN) | 185 Caught Churners (TP)

* 📈 **2. Logistic Regression (Probabilistic Baseline)**
  * **ROC-AUC**: `0.8555`
  * **Accuracy**: `80.84%`
  * **Precision**: `66.07%`
  * **Recall**: `57.22%`
  * **F1-Score**: `0.6132`
  * **Confusion Matrix**: 983 True Retained (TN) | 112 False Alarms (FP) | 160 Missed Churners (FN) | 214 Caught Churners (TP)

* 🚀 **3. Baseline XGBoost**
  * **ROC-AUC**: `0.8350`
  * **Accuracy**: `80.06%`
  * **Precision**: `64.33%`
  * **Recall**: `55.88%`
  * **F1-Score**: `0.5980`
  * **Confusion Matrix**: 920 True Retained (TN) | 115 False Alarms (FP) | 165 Missed Churners (FN) | 209 Caught Churners (TP)

* 🌲 **4. Random Forest Classifier**
  * **ROC-AUC**: `0.8327`
  * **Accuracy**: `78.92%`
  * **Precision**: `62.62%`
  * **Recall**: `51.07%`
  * **F1-Score**: `0.5626`
  * **Confusion Matrix**: 921 True Retained (TN) | 114 False Alarms (FP) | 183 Missed Churners (FN) | 191 Caught Churners (TP)

---

## 5. Model Explainability (SHAP Values)

To move beyond black-box ML models, **SHAP (SHapley Additive exPlanations)** was implemented:

- **Top Global Drivers of Churn**:
  1. `contract_Month-to-month` (+ SHAP risk contribution)
  2. `tenure` (- SHAP contribution: longer tenure reduces churn)
  3. `internet_service_Fiber optic` (+ SHAP risk contribution)
  4. `payment_method_Electronic check` (+ SHAP risk contribution)
  5. `total_services` (- SHAP contribution: more services reduce churn)

---

## 6. Software Architecture & Deployment

- **REST API (FastAPI)**: Implements `POST /predict`, `GET /health`, and `GET /api/history` with Pydantic validation schemas.
- **SQLite Prediction Logging**: Automatically logs every prediction to `predictions.db` for audit trail reporting.
- **Bootstrap 5 UI Dashboard**: Features interactive risk meters, real-time AJAX requests, and sample data loading.
- **Docker Containerization**: Includes `Dockerfile` and `docker-compose.yml` for isolated container deployment.

---

## 7. Conclusion & Future Recommendations

### Conclusion
The Customer Churn Prediction System provides an end-to-end operational pipeline. The Tuned XGBoost model achieves an ROC-AUC of 0.8568, providing accurate risk stratification to drive customer retention.

### Future Recommendations
1. **SMOTE Over-sampling**: Test Synthetic Minority Over-sampling Technique to further boost Recall for positive churn cases.
2. **Real-time CRM Integration**: Connect REST API endpoints directly with Salesforce or HubSpot CRM workflows.
3. **Model Monitoring**: Implement drift detection (Evidently AI / MLflow) to trigger automatic model retraining when customer behavior shifts.
