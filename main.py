from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Define the FastAPI app
app = FastAPI(title="Student Success Prediction API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the input data schema
class PredictionInput(BaseModel):
    study_hours: float
    attendance_rate: float
    coursework_score: float
    extracurricular_activities: int

# Load the trained model
MODEL_PATH = os.path.join("models", "model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Please train the model first.")
    
    # Prepare the input data for prediction
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return {
        "prediction": int(prediction[0]),
        "status": "Passed" if prediction[0] == 1 else "Failed",
        "probability": float(probability[0][prediction[0]])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
