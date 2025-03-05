from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load models and scaler
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# Request schemas
class ClusterPredictionRequest(BaseModel):
    distance_km: float
    delivery_time_minutes: float
    order_volume: int
    active_orders: int
    traffic_density: float

class DeliveryTimePredictionRequest(BaseModel):
    distance_km: float
    order_volume: int
    active_orders: int
    traffic_density: float
    cluster: int


# Cluster prediction endpoint
@app.post("/predict-cluster")
async def predict_cluster(data: ClusterPredictionRequest):
    features = [[
        data.distance_km, 
        data.delivery_time_minutes, 
        data.order_volume, 
        data.active_orders, 
        data.traffic_density
    ]]
    
    scaled_features = scaler.transform(features)
    cluster = int(kmeans_model.predict(scaled_features)[0])
    
    return {"predicted_cluster": cluster}


# Delivery time prediction endpoint
@app.post("/predict-delivery-time")
async def predict_delivery_time(data: DeliveryTimePredictionRequest):
    features = [[
        data.distance_km, 
        data.order_volume, 
        data.active_orders, 
        data.traffic_density, 
        data.cluster
    ]]
    
    # Directly pass the array to XGBoost without DMatrix
    predicted_time = float(xgb_model.predict(features)[0])
    
    return {"predicted_delivery_time_minutes": predicted_time}

    
    dmatrix = xgb.DMatrix(features)
    predicted_time = float(xgb_model.predict(dmatrix)[0])
    
    return {"predicted_delivery_time_minutes": predicted_time}


# Run the API with: uvicorn main:app --reload
# Test with Postman or curl 🚀
