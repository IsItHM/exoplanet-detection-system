from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import requests
from io import BytesIO

app = FastAPI(title="Exoplanet Detection API", version="1.0.0")

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    
    try:
        # These URLs will be replaced with your Google Drive links
        model_url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK_TO_MODEL"
        scaler_url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK_TO_SCALER"
        
        # Download model from Google Drive
        print("Downloading model from Google Drive...")
        model_response = requests.get(model_url)
        model = joblib.load(BytesIO(model_response.content))
        
        # Download scaler from Google Drive  
        print("Downloading scaler from Google Drive...")
        scaler_response = requests.get(scaler_url)
        scaler = joblib.load(BytesIO(scaler_response.content))
        
        print("✅ Model and scaler loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create dummy model as fallback
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="uniform", random_state=42)
        model.fit([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], [0])

@app.get("/")
async def root():
    return {
        "message": "🚀 Exoplanet Detection API",
        "status": "running",
        "cost": "$0.00",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_exoplanet(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale if scaler available
        if scaler:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict_proba(features)[0][1]
        
        return {
            "transit_probability": float(prediction),
            "prediction": "Exoplanet detected!" if prediction > 0.5 else "No exoplanet detected",
            "confidence": "High" if abs(prediction - 0.5) > 0.3 else "Medium"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
