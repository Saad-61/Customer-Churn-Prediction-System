import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.schemas import CustomerChurnInput, ChurnPredictionResponse, HealthResponse
from predict import predict_single_customer, load_prediction_pipeline
from utils.db_logger import log_prediction, get_prediction_history, init_db

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-end Machine Learning REST API for real-time customer churn prediction.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize database table
init_db()

# Cached model pipeline
PIPELINE = None

def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        try:
            PIPELINE = load_prediction_pipeline()
        except FileNotFoundError:
            PIPELINE = None
    return PIPELINE

@app.on_event("startup")
def startup_event():
    get_pipeline()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders enterprise SaaS web dashboard.
    """
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API Health check endpoint.
    """
    pipeline = get_pipeline()
    if pipeline is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=pipeline.get("model_name", "Tuned XGBoost")
        )
    return HealthResponse(
        status="degraded (model artifact missing - run python train.py)",
        model_loaded=False
    )

@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(payload: CustomerChurnInput):
    """
    POST /predict
    Accepts customer profile JSON, applies feature engineering, transforms data,
    predicts churn probability, and logs prediction record to SQLite.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please execute `python train.py` first to train and persist the model."
        )
        
    try:
        data_dict = payload.dict()
        res = predict_single_customer(data_dict, pipeline)
        
        # Bonus task: Log to SQLite database
        record_id = log_prediction(
            data=data_dict,
            predicted_churn=res["predicted_churn"],
            churn_probability=res["churn_probability"]
        )
        
        res["logged_db_id"] = record_id
        return ChurnPredictionResponse(**res)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/history")
async def fetch_history(limit: int = 15):
    """
    Returns recent SQLite prediction logs.
    """
    history = get_prediction_history(limit=limit)
    return {"status": "success", "count": len(history), "data": history}

@app.get("/api/metrics")
async def fetch_metrics():
    """
    Returns model training comparison metrics.
    """
    metrics_path = os.path.join("data", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return {"status": "success", "metrics": data}
    return {"status": "error", "message": "No training metrics found."}
