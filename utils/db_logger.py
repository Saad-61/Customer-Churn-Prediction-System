import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any

DB_FILE = "predictions.db"

def init_db(db_path: str = DB_FILE):
    """
    Initializes SQLite database table for prediction logging.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS churn_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            customer_id TEXT,
            tenure INTEGER,
            contract TEXT,
            monthly_charges REAL,
            total_charges REAL,
            internet_service TEXT,
            payment_method TEXT,
            predicted_churn INTEGER,
            churn_probability REAL,
            risk_level TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(data: Dict[str, Any], predicted_churn: int, churn_probability: float, db_path: str = DB_FILE) -> int:
    """
    Logs customer parameters and prediction result to SQLite database.
    """
    init_db(db_path)
    
    # Calculate risk level category
    if churn_probability >= 0.70:
        risk_level = "High Risk"
    elif churn_probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO churn_predictions (
            timestamp, customer_id, tenure, contract, monthly_charges, total_charges,
            internet_service, payment_method, predicted_churn, churn_probability, risk_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(data.get("customer_id", "DEMO-USER")),
        int(data.get("tenure", 0)),
        str(data.get("contract", "Month-to-month")),
        float(data.get("monthly_charges", 0.0)),
        float(data.get("total_charges", 0.0)),
        str(data.get("internet_service", "Fiber optic")),
        str(data.get("payment_method", "Electronic check")),
        int(predicted_churn),
        round(float(churn_probability), 4),
        risk_level
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_prediction_history(limit: int = 20, db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    """
    Fetches recent prediction records from SQLite database.
    """
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, customer_id, tenure, contract, monthly_charges,
               total_charges, internet_service, payment_method, predicted_churn,
               churn_probability, risk_level
        FROM churn_predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
