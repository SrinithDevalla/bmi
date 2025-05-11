import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess
df = pd.read_csv("cleaned_gps_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.sort_values("Timestamp", inplace=True)

# Normalize
traffic_series = df["Traffic_Density"].values.reshape(-1, 1)
scaler = MinMaxScaler()
traffic_series_scaled = scaler.fit_transform(traffic_series)

# Sequence creation
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_len = 10
X_seq, y_seq = create_sequences(traffic_series_scaled, seq_len)

# Train/Test split
X_train, X_test = X_seq[:150], X_seq[150:]
y_train, y_test = y_seq[:150], y_seq[150:]

# Reshape for LSTM
X_train = X_train.reshape((-1, seq_len, 1))
X_test = X_test.reshape((-1, seq_len, 1))

# Build model
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

print("✅ LSTM Forecasting Model Trained")


# Predict on test set and inverse transform
predicted = model_lstm.predict(X_test)
predicted_inverse = scaler.inverse_transform(predicted)
actual_inverse = scaler.inverse_transform(y_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(actual_inverse, label="Actual")
plt.plot(predicted_inverse, label="Predicted")
plt.title("📉 LSTM Traffic Density Forecast")
plt.xlabel("Time Steps")
plt.ylabel("Traffic Density")
plt.legend()
plt.savefig("static/forecast_plot.png")
