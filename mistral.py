from fastapi import FastAPI
from pydantic import BaseModel
from geopy.distance import geodesic
import google.generativeai as genai
import mysql.connector
import joblib
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()


# Configure Gemini API
genai.configure(api_key='AIzaSyCeIxjSIV8iTu6FXiPOzFaTZyoI3YO_iVM')  # Replace with your Gemini API key
model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp-1219')

app = FastAPI()

class UserLocation(BaseModel):
    latitude: float
    longitude: float
    n_clusters: int
    min_distance_km: float

# MySQL connection
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "4102",
    "database": "pot"
}
kmeans_model = joblib.load("kmeans_model.pkl")

def get_nearby_stores(lat, lng, radius):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT id, name, latitude, longitude, capacity, orders_served, traffic_density, historical_avg_delivery_time,
        (6371 * ACOS(COS(RADIANS(%s)) * COS(RADIANS(latitude)) * 
        COS(RADIANS(longitude) - RADIANS(%s)) + 
        SIN(RADIANS(%s)) * SIN(RADIANS(latitude)))) AS distance
    FROM store
    HAVING distance <= %s
    ORDER BY distance;
    """

    cursor.execute(query, (lat, lng, lat, radius))
    stores = cursor.fetchall()
    conn.close()

    return stores

@app.post("/analyze-impact")
async def analyze_impact(user_loc: UserLocation):
    stores = get_nearby_stores(user_loc.latitude, user_loc.longitude, user_loc.min_distance_km)

    # Calculate delivery times for nearby stores
    predictions = []
    for store in stores:
        predicted_time = store['historical_avg_delivery_time'] + (store['traffic_density'] * 5)
        predictions.append({"store_id": store['id'], "predicted_time": predicted_time})

    # Use clustering to suggest new store locations
    store_coords = [(store['latitude'], store['longitude']) for store in stores]

    kmeans_model.set_params(n_clusters=user_loc.n_clusters)
    kmeans_model.fit(store_coords)
    new_store_coords = kmeans_model.cluster_centers_.tolist()

    # Analyze store load and make recommendations
    analyzed_stores = []
    for store in stores:
        recommendation = ""

        if store['traffic_density'] > 7 and store['orders_served'] >= 0.9 * store['capacity']:
            recommendation = "🚩 High demand! Opening a new store nearby can reduce delivery time and handle overflow orders."
        elif store['traffic_density'] < 3:
            recommendation = "🟢 Low traffic area — might not need a new store immediately."
        else:
            recommendation = "🔍 Moderate demand — consider opening a store if future growth is expected."

        analyzed_stores.append({
            **store,
            "recommendation": recommendation
        })

    # Generate business insights with Gemini
    insights_prompt = f"""
    A user clicked on coordinates ({user_loc.latitude}, {user_loc.longitude}).

    1. Identify the optimal new locations to open dark stores within a radius of {user_loc.min_distance_km} km.
    2. Provide the suggested new store coordinates: {new_store_coords}.
    3. Analyze how these new stores will affect the following metrics for the 3 nearest existing dark stores:
       - Delivery times: {predictions}
       - Load distribution (capacity vs. orders served)
       - Traffic density impact
    4. Offer a detailed business-level summary outlining potential improvements in delivery speed, store load balancing, and customer satisfaction.
    5. Suggest strategic locations that reduce delivery bottlenecks and improve operational efficiency.
    6. Evaluate population density and traffic trends to justify new store openings.
    """

    insight_response = model.generate_content(insights_prompt)
    business_insights = insight_response.text

    return {
        "nearby_stores": analyzed_stores,
        "predicted_delivery_times": predictions,
        "suggested_new_store_locations": new_store_coords,
        "business_insights": business_insights
    }

@app.get("/nearby-stores")
async def get_stores(lat: float, lng: float, radius: float):
    stores = get_nearby_stores(lat, lng, radius)

    analyzed_stores = []
    for store in stores:
        recommendation = ""

        if store['traffic_density'] > 7 and store['orders_served'] >= 0.9 * store['capacity']:
            recommendation = "🚩 High demand! Opening a new store nearby can reduce delivery time and handle overflow orders."
        elif store['traffic_density'] < 3:
            recommendation = "🟢 Low traffic area — might not need a new store immediately."
        else:
            recommendation = "🔍 Moderate demand — consider opening a store if future growth is expected."

        analyzed_stores.append({
            **store,
            "recommendation": recommendation
        })

    return {"analyzed_stores": analyzed_stores}
