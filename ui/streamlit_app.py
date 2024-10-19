import streamlit as st
import requests
import joblib
import numpy as np
#from sklearn.preprocessing import StandardScaler

# Load the saved scaler and model
scaler = joblib.load('../models/scaler.pkl')  # Load the scaler used in model training
model = joblib.load('../models/random_forest_churn_model.pkl')  # Load the trained model

# Streamlit UI elements for user input (raw values)
st.title("Customer Churn Prediction")

# Raw inputs from user (not scaled)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=619)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=42)  # Actual age, not scaled
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Account Balance", value=50000.0)  # Actual balance, not scaled
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=60000.0)  # Actual salary, not scaled

# Country dropdown and one-hot encoding
country = st.selectbox("Country", ["France", "Germany", "Spain"])

# One-hot encoding for country
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0
# France is the default, so no need to create a feature for it (both country_Germany and country_Spain will be 0)

# Convert categorical features (gender, credit_card, active_member) to numerical
gender = 1 if gender == "Male" else 0
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

# Only scale the features that were scaled during training (balance, age, estimated_salary)
raw_input_data = np.array([[balance, age, estimated_salary]])  # Shape the raw data for scaling
scaled_features = scaler.transform(raw_input_data)[0]  # Apply scaling using the trained scaler

# Combine all features into a list (scaled features + other raw features)
input_data = [
    credit_score,  # credit score (raw, not scaled)
    gender,        # gender (1 for male, 0 for female)
    scaled_features[1],  # scaled age
    tenure,        # tenure (raw, not scaled)
    scaled_features[0],  # scaled balance
    products_number,  # number of products (raw, not scaled)
    credit_card,   # credit card (1 for yes, 0 for no)
    active_member,  # active member (1 for yes, 0 for no)
    scaled_features[2],  # scaled estimated salary
    country_Germany,   # one-hot encoded country (Germany)
    country_Spain      # one-hot encoded country (Spain)
]

if st.button("Predict Churn"):
    # Send the processed data to the Flask API for prediction
    response = requests.post('http://127.0.0.1:5000/predict', json={'features': input_data})
    
    if response.status_code == 200:
        prediction = response.json()
        if prediction['churn'] == 1:
            st.error(f"The customer is likely to churn. Probability: {prediction['churn_probability']:.2f}")
        else:
            st.success(f"The customer is not likely to churn. Probability: {prediction['churn_probability']:.2f}")
    else:
        st.error("Failed to get prediction")
