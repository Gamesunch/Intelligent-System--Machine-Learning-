import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the trained model
model = load_model("models/Nvidia_model.h5")

st.title("Nvidia Stock Price Prediction 🤖")

st.markdown("""
<style>
.big-font {
    font-size:24px !important;
}
            
.medium-font {
    font-size:20px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<p class="big-font">
This application allows you to predict the future stock prices of Nvidia using a trained LSTM model.  
<br>  
You can input the past 60 days of stock prices to predict the next day's price.
</p>
""", unsafe_allow_html=True)


# Function to generate random stock prices for the past 60 days
def generate_random_prices():
    # Generate 60 random stock prices within a reasonable range (e.g., 350 to 650)
    return np.round(np.random.uniform(350, 650, 60), 2)

# Button to generate random input
if st.button('Generate Random 60-Day Stock Prices'):
    random_prices = generate_random_prices()
    input_data = ",".join(map(str, random_prices))
    st.text_area("Enter the last 60 days of Nvidia stock prices (comma separated):", value=input_data, height=200)

# Input: Allow the user to input the last 60 days of stock prices
input_data = st.text_area(
    "Enter the last 60 days of Nvidia stock prices (comma separated):",
    "350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635"
)

# Convert input string to a list of floats
if input_data:
    try:
        input_list = list(map(float, input_data.split(',')))
        
        # Check if the input data has 60 values
        if len(input_list) == 60:
            # Normalize the input data
            scaler = MinMaxScaler(feature_range=(0, 1))
            input_scaled = scaler.fit_transform(np.array(input_list).reshape(-1, 1))

            # Reshape input for prediction (1, time_steps, 1)
            X_input = input_scaled.reshape((1, 60, 1))

            # Predict using the trained model
            predicted_price = model.predict(X_input)

            # Convert the prediction back to the original scale
            predicted_price = scaler.inverse_transform(predicted_price)

            # Display predicted price for the next day
            st.markdown('<p class="big-font">Prediction for the Next Day</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="medium-font">Predicted Price: ${predicted_price[-1][0]:.2f}</p>', unsafe_allow_html=True)

        else:
            st.error("Please enter exactly 60 values.")
    except ValueError:
        st.error("Invalid input. Please ensure you are entering valid numbers separated by commas.")

# Show the actual vs. predicted price data
st.subheader("Actual vs Predicted Stock Prices")

df = pd.read_csv("data/NVDA.csv")

# Convert Date column to datetime and sort
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Use the last 60 days for predictions
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

time_steps = 60

# Create sequences
def create_sequences(data, time_steps=60):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
    return np.array(X)

X_test = create_sequences(prices_scaled, time_steps)

# Predict prices
predicted_prices = model.predict(X_test)

# Convert predictions back to the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)

# Ensure date alignment
actual_dates = df["Date"].iloc[time_steps:]

# Plot actual vs predicted prices
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(actual_dates, df["Close"].iloc[time_steps:], label="Actual Price", color='blue')
ax.plot(actual_dates, predicted_prices, label="Predicted Price", color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("Actual vs Predicted Nvidia Stock Prices")
ax.legend()
st.pyplot(fig)

# Show Training & Validation Loss
st.subheader("Training & Validation Loss")

try:
    with open("models/training_history.pkl", "rb") as f:
        history = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["loss"], label="Training Loss", color="blue")
    ax.plot(history["val_loss"], label="Validation Loss", color="red")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("Training history file not found. Train the model first.")
