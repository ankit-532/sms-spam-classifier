# src/main.py
# FastAPI application

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import SMSRequest, SMSResponse
from src.predictor import predict_sms, load_model

app = FastAPI(
    title="SMS Spam Classifier API",
    description="AI-powered SMS spam detection with hybrid classification",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"service": "SMS Spam Classifier API", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=SMSResponse)
async def predict(request: SMSRequest):
    try:
        result = predict_sms(request.message)
        return SMSResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))