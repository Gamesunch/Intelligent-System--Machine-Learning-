import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the trained model
model = load_model("models/Nvidia_model.h5")

# Streamlit UI Configuration
st.set_page_config(page_title="Nvidia Stock Price Prediction", layout="wide")

# Title and Description
st.title("📈 Nvidia Stock Price Prediction")
st.markdown("""
Welcome to the **Nvidia Stock Price Predictor**! 🚀

โดยเดโม่ตัวนี้จะให้คุณทำนายราคาหุ้นของ Nvidia ในวันถัดไป โดยให้คุณใส่ราคาหุ้นของ Nvidia ใน 60 วันที่ผ่านมา

👉 ใส่ราคาหุุ้นเมื่อ 60 วันที่ผ่านมา
""", unsafe_allow_html=True)

# Sidebar for Input
st.sidebar.header("🔢 ใส่ Data")

def generate_random_prices():
    return np.round(np.random.uniform(350, 650, 60), 2)

# Initialize session state for input field
if "input_data" not in st.session_state:
    st.session_state.input_data = "350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635"

# Generate Random Data Button
if st.sidebar.button("🎲 Generate Random Data"):
    st.session_state.input_data = ",".join(map(str, generate_random_prices()))

# Input Text Area
input_data = st.sidebar.text_area("ใส่ราคาหุ้น Nvidia ใน 60 วัน (เว้นด้วยคอมม่า Ex 123,456,123):", 
                                  st.session_state.input_data, height=150)

# Update session state with user input
st.session_state.input_data = input_data

# Predict Button
if st.sidebar.button("🔮 ทำนายยยยยยย"):
    st.session_state.predict = True
else:
    st.session_state.predict = False

# Prediction Section
st.header("📊 ผลลัพธ์การทำนาย")

if st.session_state.predict:
    try:
        input_list = list(map(float, st.session_state.input_data.split(',')))
        
        if len(input_list) == 60:
            scaler = MinMaxScaler(feature_range=(0, 1))
            input_scaled = scaler.fit_transform(np.array(input_list).reshape(-1, 1))
            X_input = input_scaled.reshape((1, 60, 1))
            predicted_price = model.predict(X_input)
            predicted_price = scaler.inverse_transform(predicted_price)
            
            st.success(f"📉 Predicted Stock Price for Next Day: **${predicted_price[-1][0]:.2f}**")
        else:
            st.error("⚠️ Please enter exactly 60 values.")
    except ValueError:
        st.error("❌ Invalid input! Please enter valid numbers separated by commas.")

# Actual vs Predicted Stock Prices
st.header("📈 Actual vs Predicted Stock Prices")
df = pd.read_csv("data/NVDA.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

def create_sequences(data, time_steps=60):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
    return np.array(X)

X_test = create_sequences(prices_scaled, 60)
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_dates = df["Date"].iloc[60:]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(actual_dates, df["Close"].iloc[60:], label="Actual Price", color='blue', linewidth=2)
ax.plot(actual_dates, predicted_prices, label="Predicted Price", color='red', linestyle="dashed", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("Actual vs Predicted Nvidia Stock Prices")
ax.legend()
st.pyplot(fig)

# Training & Validation Loss
st.header("📉 Training & Validation Loss")
try:
    with open("models/training_history.pkl", "rb") as f:
        history = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["loss"], label="Training Loss", color="blue", linewidth=2)
    ax.plot(history["val_loss"], label="Validation Loss", color="red", linestyle="dashed", linewidth=2)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("⚠️ Training history file not found. Please retrain the model.")