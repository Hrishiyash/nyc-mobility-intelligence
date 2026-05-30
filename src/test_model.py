import joblib
import pandas as pd

model = joblib.load("models/xgboost_trip_duration.pkl")

sample = pd.DataFrame([{
    "passenger_count": 1,
    "trip_distance": 5.0,
    "RatecodeID": 1,
    "PULocationID": 100,
    "DOLocationID": 150,
    "payment_type": 1,
    "pickup_hour": 10,
    "pickup_day": 15,
    "pickup_weekday": 2,
    "pickup_month": 1
}])

prediction = model.predict(sample)

print("Predicted Duration:",prediction[0])