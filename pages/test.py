import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained models and scaler
svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset to get feature names
df = pd.read_csv("data/Life Expectancy Data.csv")
df.columns = df.columns.str.strip()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Country", "Year", "Life expectancy"], errors="ignore")
feature_columns = X.columns

# Streamlit UI
st.title("Life Expectancy Prediction")
st.write("Enter values below or generate random data for predictions:")

# Function to generate random input data
def generate_random_data():
    return {col: np.random.uniform(X[col].min(), X[col].max()) for col in feature_columns}

# Ensure session state exists for random data
if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_data()

# Button to regenerate random data
if st.button("Generate Random Data"):
    st.session_state.random_data = generate_random_data()

# Use session state random data
random_data = st.session_state.random_data

# User inputs for model prediction
input_values = {col: st.number_input(col, value=random_data[col]) for col in feature_columns}

# Collect inputs into a DataFrame
input_data = pd.DataFrame([input_values])

# Scale the input data properly
input_scaled = scaler.transform(input_data)

# Debugging - Print Input Data Before Prediction
st.write("🔍 Input Data Before Scaling:", input_data)
st.write("🔍 Scaled Input Data:", input_scaled)

# Predict
if st.button("Predict"):
    st.session_state.svm_prediction = svm_model.predict(input_scaled)[0]
    st.session_state.rf_prediction = rf_model.predict(input_scaled)[0]
    st.session_state.prediction_made = True  # Track if predictions were made


# Display results
if "svm_prediction" in st.session_state:
    st.subheader("Predictions")
    st.write(f"🌟 SVM Model Prediction: **{st.session_state.svm_prediction:.2f}** years")
    st.write(f"🌟 Random Forest Model Prediction: **{st.session_state.rf_prediction:.2f}** years")
    
    # Load model performance data
    results_df = pd.read_csv("TrainModel/model_performance.csv")
    st.subheader("Model Performance")
    st.write(results_df)
    
    # Graph
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["Model"], results_df["R2 Score"], color=["blue", "green"])
    ax.set_title("Model Comparison: R2 Score")
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
