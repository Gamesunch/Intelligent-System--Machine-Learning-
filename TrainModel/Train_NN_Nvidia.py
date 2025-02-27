import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle  # To save training history
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("NVDA.csv")

# Convert Date column to datetime and sort
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Select feature for LSTM (we predict "Close" price)
prices = df["Close"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60  
X, y = create_sequences(prices_scaled, time_steps)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),  # Prevents overfitting
    LSTM(50),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])


# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model and store history
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Predict on test data
predicted_prices = model.predict(X_test)

# Convert predictions back to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot actual vs predicted prices
plt.figure(figsize=(14, 6))
plt.plot(df["Date"].iloc[-len(y_test):], df["Close"].iloc[-len(y_test):], label="Actual Price", color='blue')
plt.plot(df["Date"].iloc[-len(y_test):], predicted_prices, label="Predicted Price", color='red')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Plot training loss and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label="Training Loss", color='blue')
plt.plot(history.history['val_loss'], label="Validation Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Evaluate the model
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), predicted_prices))
print(f"RMSE: {rmse}")

# Save the model
model.save('Nvidia_model.h5')
