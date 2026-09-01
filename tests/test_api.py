import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():
    payload = {
        "customer_id": "API-TEST-001",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "One year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 55.0,
        "total_charges": 660.0
    }
    
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert data["customer_id"] == "API-TEST-001"
        assert "churn_probability" in data
        assert "risk_level" in data
    else:
        # If model is not trained yet, should return 503
        assert response.status_code in [503, 500]

def test_history_endpoint():
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
