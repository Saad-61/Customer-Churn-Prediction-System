from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerChurnInput(BaseModel):
    customer_id: Optional[str] = Field(default="CUST-001", description="Unique identifier for the customer")
    gender: str = Field(default="Female", description="Gender of the customer (Female/Male)")
    senior_citizen: int = Field(default=0, description="Whether the customer is a senior citizen (1/0)")
    partner: str = Field(default="No", description="Whether customer has a partner (Yes/No)")
    dependents: str = Field(default="No", description="Whether customer has dependents (Yes/No)")
    tenure: int = Field(default=2, description="Number of months customer has stayed with company")
    phone_service: str = Field(default="Yes", description="Whether customer has phone service (Yes/No)")
    multiple_lines: str = Field(default="No", description="Whether customer has multiple lines (Yes/No/No phone service)")
    internet_service: str = Field(default="Fiber optic", description="Type of internet service (DSL/Fiber optic/No)")
    online_security: str = Field(default="No", description="Online security add-on (Yes/No/No internet service)")
    online_backup: str = Field(default="No", description="Online backup add-on (Yes/No/No internet service)")
    device_protection: str = Field(default="No", description="Device protection add-on (Yes/No/No internet service)")
    tech_support: str = Field(default="No", description="Tech support add-on (Yes/No/No internet service)")
    streaming_tv: str = Field(default="Yes", description="Streaming TV service (Yes/No/No internet service)")
    streaming_movies: str = Field(default="Yes", description="Streaming movies service (Yes/No/No internet service)")
    contract: str = Field(default="Month-to-month", description="Contract term (Month-to-month/One year/Two year)")
    paperless_billing: str = Field(default="Yes", description="Paperless billing option (Yes/No)")
    payment_method: str = Field(default="Electronic check", description="Payment method used")
    monthly_charges: float = Field(default=89.85, description="Current monthly charge amount")
    total_charges: float = Field(default=179.70, description="Total charges billed to date")

class ChurnPredictionResponse(BaseModel):
    status: str = "success"
    customer_id: str
    predicted_churn: int
    churn_label: str
    churn_probability: float
    churn_probability_percent: float
    risk_level: str
    recommendation: str
    model_used: str
    logged_db_id: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
