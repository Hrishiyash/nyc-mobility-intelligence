from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Load model once when API starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "xgboost_trip_duration.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None



class TripInput(BaseModel):
    passenger_count: int
    trip_distance: float
    RatecodeID: float
    PULocationID: int
    DOLocationID: int
    payment_type: int
    pickup_hour: int
    pickup_day: int
    pickup_weekday: int
    pickup_month: int


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: TripInput):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    input_df = pd.DataFrame([{
        "passenger_count": data.passenger_count,
        "trip_distance": data.trip_distance,
        "RatecodeID": data.RatecodeID,
        "PULocationID": data.PULocationID,
        "DOLocationID": data.DOLocationID,
        "payment_type": data.payment_type,
        "pickup_hour": data.pickup_hour,
        "pickup_day": data.pickup_day,
        "pickup_weekday": data.pickup_weekday,
        "pickup_month": data.pickup_month
    }])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_duration_minutes": round(float(prediction), 2)
    }