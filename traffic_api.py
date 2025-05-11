import requests

API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
origin = "12.9716,77.5946"
destination = "12.9260,77.6762"

url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&departure_time=now&traffic_model=best_guess&key={API_KEY}"
response = requests.get(url)
distance_data = response.json()

print(" Live Distance Data Fetched")

