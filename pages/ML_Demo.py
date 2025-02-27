import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained models and scaler
svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")  # Load pre-fitted scaler

st.write("✅ SVM Model Loaded:", svm_model)

# Load dataset to get feature names
df = pd.read_csv("data/Life Expectancy Data.csv")
df_clean = df.drop(columns=["Country", "Year"], errors="ignore")
df_clean["Status"] = df_clean["Status"].astype("category").cat.codes
X = df_clean.drop(columns=["Life expectancy "], errors="ignore")

# Ensure correct feature order for input
feature_columns = X.columns  

# Streamlit UI
st.title("Life Expectancy Prediction")
st.write("Enter values below or generate random data for predictions:")

# Function to generate random input data
def generate_random_data():
    return {
        "Adult Mortality": np.random.uniform(0, 500),
        "Infant Deaths": float(np.random.randint(0, 100)),
        "Alcohol": np.random.uniform(0, 15),
        "Percentage Expenditure": np.random.uniform(0, 5000),
        "Hepatitis B": float(np.random.randint(0, 100)),
        "Measles": float(np.random.randint(0, 50000)),
        "BMI": np.random.uniform(10, 50),
        "Under Five Deaths": float(np.random.randint(0, 200)),
        "Polio": float(np.random.randint(0, 100)),
        "Total Expenditure": np.random.uniform(0, 20),
        "Diphtheria": float(np.random.randint(0, 100)),
        "HIV/AIDS": np.random.uniform(0, 10),
        "GDP": np.random.uniform(500, 100000),
        "Population": float(np.random.randint(1000, 100000000)),
        "Thinness 1-19 years": np.random.uniform(0, 20),
        "Thinness 5-9 years": np.random.uniform(0, 20),
        "Income Composition of Resources": np.random.uniform(0, 1),
        "Schooling": np.random.uniform(1, 20),
        "Status": float(np.random.randint(0, 2)),  # Encoded as 0 or 1
    }

# Ensure session state exists for random data
if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_data()

# Button to regenerate random data
if st.button("Generate Random Data"):
    st.session_state.random_data = generate_random_data()

# Use session state random data
random_data = st.session_state.random_data

# User inputs for model prediction
col1, col2 = st.columns(2)
with col1:
    adult_mortality = st.number_input("Adult Mortality", min_value=0.0, step=1.0, value=random_data["Adult Mortality"])
    infant_deaths = st.number_input("Infant Deaths", min_value=0.0, step=1.0, value=random_data["Infant Deaths"])
    alcohol = st.number_input("Alcohol Consumption", min_value=0.0, step=0.1, value=random_data["Alcohol"])
    percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0, step=0.1, value=random_data["Percentage Expenditure"])
    hepatitis_B = st.number_input("Hepatitis B", min_value=0.0, step=1.0, value=random_data["Hepatitis B"])
    measles = st.number_input("Measles", min_value=0.0, step=1.0, value=random_data["Measles"])
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=random_data["BMI"])
    under_five_deaths = st.number_input("Under Five Deaths", min_value=0.0, step=1.0, value=random_data["Under Five Deaths"])
    polio = st.number_input("Polio", min_value=0.0, step=1.0, value=random_data["Polio"])
    total_expenditure = st.number_input("Total Expenditure", min_value=0.0, step=0.1, value=random_data["Total Expenditure"])

with col2:
    diphtheria = st.number_input("Diphtheria", min_value=0.0, step=1.0, value=random_data["Diphtheria"])
    hiv_aids = st.number_input("HIV/AIDS", min_value=0.0, step=0.1, value=random_data["HIV/AIDS"])
    gdp = st.number_input("GDP", min_value=0.0, step=1.0, value=random_data["GDP"])
    population = st.number_input("Population", min_value=0.0, step=1.0, value=random_data["Population"])
    thinness_1_19 = st.number_input("Thinness 1-19 years", min_value=0.0, step=0.1, value=random_data["Thinness 1-19 years"])
    thinness_5_9 = st.number_input("Thinness 5-9 years", min_value=0.0, step=0.1, value=random_data["Thinness 5-9 years"])
    income_composition = st.number_input("Income Composition of Resources", min_value=0.0, step=0.01, value=random_data["Income Composition of Resources"])
    schooling = st.number_input("Schooling", min_value=0.0, step=0.1, value=random_data["Schooling"])
    status = st.number_input("Status (0: Developing, 1: Developed)", min_value=0, max_value=1, step=1, value=int(random_data["Status"]))

# Collect inputs into a DataFrame to preserve feature names
input_data = pd.DataFrame([[  
    adult_mortality, infant_deaths, alcohol, percentage_expenditure,
    hepatitis_B, measles, bmi, under_five_deaths, polio,
    total_expenditure, diphtheria, hiv_aids, gdp, population,
    thinness_1_19, thinness_5_9, income_composition, schooling, status
]], columns=feature_columns)  

# Try skipping scaling
input_scaled = input_data.to_numpy()  # Convert to NumPy array

# Before prediction
st.write("Input shape:", input_scaled.shape)
st.write("Input sample:", input_scaled[0:1])

# Ensure session state exists for predictions
if "svm_prediction" not in st.session_state:
    st.session_state.svm_prediction = None
if "rf_prediction" not in st.session_state:
    st.session_state.rf_prediction = None

# Make predictions when button is clicked
if st.button("Predict"):
    st.session_state.svm_prediction = svm_model.predict(input_scaled)[0]
    st.session_state.rf_prediction = rf_model.predict(input_scaled)[0]
    st.session_state.prediction_made = True  # Track if predictions were made


# Display results only if predictions exist
if st.session_state.get("prediction_made", False):
    st.subheader("Predictions")
    st.write(f"🌟 SVM Model Prediction: **{st.session_state.svm_prediction:.2f}** years")
    st.write(f"🌟 Random Forest Model Prediction: **{st.session_state.rf_prediction:.2f}** years")


    # Display model comparison
    results_df = pd.read_csv("TrainModel/model_performance.csv")
    st.bar_chart(results_df.set_index("Model"))
