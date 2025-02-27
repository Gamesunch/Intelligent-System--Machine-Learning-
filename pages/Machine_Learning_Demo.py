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

# Streamlit UI Configuration
st.set_page_config(page_title="Life Expectancy Prediction", layout="wide")

# Title and Description
st.title("🌍 Life Expectancy Prediction")
st.markdown("""
Welcome to the **Life Expectancy Predictor**! 🏥📊

This tool uses **SVM and Random Forest models** to predict life expectancy based on key health and economic indicators.

👉 Enter values manually or **generate random data** for predictions!
""", unsafe_allow_html=True)

# Sidebar for Input
st.sidebar.header("🔢 Input Data")

def generate_random_data():
    return {col: np.random.uniform(X[col].min(), X[col].max()) for col in feature_columns}

col1, col2 = st.sidebar.columns(2)

if col1.button("🎲 Generate Random Data"):
    st.session_state.random_data = generate_random_data()

if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_data()

random_data = st.session_state.random_data

# User inputs for model prediction
st.sidebar.subheader("Enter Data Manually")
input_values = {col: st.sidebar.number_input(col, value=random_data[col]) for col in feature_columns}
input_data = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_data)

# Debugging - Print Input Data Before Prediction
st.sidebar.write("🔍 Input Data Before Scaling:", input_data)
st.sidebar.write("🔍 Scaled Input Data:", input_scaled)

# Predict Button
if col2.button("🚀 Predict"):
    st.session_state.svm_prediction = svm_model.predict(input_scaled)[0]
    st.session_state.rf_prediction = rf_model.predict(input_scaled)[0]
    st.session_state.prediction_made = True

# Display Results
st.header("📊 Prediction Results")
if "svm_prediction" in st.session_state:
    st.success(f"🌟 **SVM Prediction:** {st.session_state.svm_prediction:.2f} years")
    st.success(f"🌲 **Random Forest Prediction:** {st.session_state.rf_prediction:.2f} years")
    
    # Load model performance data
    results_df = pd.read_csv("TrainModel/model_performance.csv")
    st.subheader("📈 Model Performance")
    st.write(results_df)
    
    # Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results_df["Model"], results_df["R2 Score"], color=["blue", "green"], alpha=0.7)
    ax.set_title("Model Comparison: R2 Score")
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
