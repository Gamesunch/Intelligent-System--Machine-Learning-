import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For better visuals
import pickle  
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("NVDA.csv")

# 📊 Data Analysis
print(df.info())  # Check data types and missing values
print(df.describe())  # Statistical summary
print("Missing values:\n", df.isnull().sum())  # Count missing values
print("Duplicates:", df.duplicated().sum())  # Check for duplicates

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")  # Sort by date

# Fill missing values
df.fillna(method="ffill", inplace=True)

# Keep only necessary columns
df = df[["Date", "Close"]]

# 📌 Data Analysis Graphs
plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Close"], label="Stock Price", color="blue")
plt.title("Nvidia Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# 📊 Stock Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Close"], bins=50, kde=True, color="purple")
plt.title("Stock Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.show()

# 📊 Moving Averages (SMA & EMA)
df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day SMA
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()  # 50-day EMA

plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Close"], label="Stock Price", color="blue", alpha=0.6)
plt.plot(df["Date"], df["SMA_50"], label="50-day SMA", color="red", linestyle="dashed")
plt.plot(df["Date"], df["EMA_50"], label="50-day EMA", color="green", linestyle="dashed")
plt.title("Stock Price with SMA & EMA")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# 📊 Box Plot for Outlier Detection
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Close"], color="orange")
plt.title("Stock Price Outlier Detection")
plt.ylabel("Price ($)")
plt.show()

# 📉 Data Preprocessing (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Function to create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences of 60 days
time_steps = 60
X, y = create_sequences(prices_scaled, time_steps)

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Make predictions on test data
predicted_prices = model.predict(X_test)

# Convert predictions back to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"RMSE: {rmse:.2f}")

# Save the trained model
model.save("Nvidia_model.h5")

# 📊 Final Graphs
plt.figure(figsize=(14, 6))
plt.plot(df["Date"].iloc[-len(y_test):], df["Close"].iloc[-len(y_test):], label="Actual Price", color='blue')
plt.plot(df["Date"].iloc[-len(y_test):], predicted_prices, label="Predicted Price", color='red', linestyle="dashed")
plt.title("Actual vs Predicted Nvidia Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# 📉 Training vs Validation Loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label="Training Loss", color="blue")
plt.plot(history.history['val_loss'], label="Validation Loss", color="red", linestyle="dashed")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 📊 Prediction Error Distribution
error = actual_prices - predicted_prices
plt.figure(figsize=(10, 5))
sns.histplot(error, bins=50, kde=True, color="red")
plt.title("Prediction Error Distribution")
plt.xlabel("Error ($)")
plt.ylabel("Frequency")
plt.show()
