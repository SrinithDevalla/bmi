import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
df = pd.read_csv("gps_data_advanced.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.drop_duplicates(inplace=True)
df.to_csv("cleaned_gps_data.csv", index=False)

# Heatmap
map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
traffic_map = folium.Map(location=map_center, zoom_start=12)
heat_data = list(zip(df["Latitude"], df["Longitude"], df["Traffic_Density"]))
HeatMap(heat_data).add_to(traffic_map)
traffic_map.save("static/traffic_heatmap.html")

# Peak traffic
df["Hour_of_Day"] = df["Timestamp"].dt.hour
traffic_by_hour = df.groupby("Hour_of_Day")["Traffic_Density"].mean()

plt.figure(figsize=(10, 5))
plt.plot(traffic_by_hour.index, traffic_by_hour.values, marker="o")
plt.xlabel("Hour")
plt.ylabel("Average Density")
plt.title(" Peak Traffic Hours")
plt.grid()
plt.savefig("static/traffic_trend.png")
plt.close()

# Clustering
features = df[["Latitude", "Longitude", "Speed", "Traffic_Density"]]
features_scaled = StandardScaler().fit_transform(features)
df["Traffic_Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(features_scaled)

# Anomalies
df["Anomaly"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(features_scaled)
anomalies = df[df["Anomaly"] == -1]

# Random Forest Prediction
X = df[["Hour_of_Day", "Speed", "Traffic_Cluster"]]
y = df["Traffic_Density"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ RF Prediction MAE: {mae:.2f}")

# Identify flyover-worthy area
high_density = df[df["Traffic_Density"] >= df["Traffic_Density"].quantile(0.95)]
flyovers = high_density.groupby(["Latitude", "Longitude"]).size().reset_index(name="count")
flyovers.to_csv("flyover_suggestions.csv", index=False)


# ROAD WIDENING LOGIC 
df["Rolling_Avg"] = df["Traffic_Density"].rolling(window=5).mean()

# Identify locations where rolling avg is above 90th percentile
road_widen_df = df[df["Rolling_Avg"] >= df["Rolling_Avg"].quantile(0.90)]

# Group by location
widening_locations = road_widen_df.groupby(["Latitude", "Longitude"]).size().reset_index(name="count")
widening_locations.to_csv("road_widening_suggestions.csv", index=False)
