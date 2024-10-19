import requests
import joblib
import numpy as np

# Load the saved scaler (assuming it is locally available)
scaler = joblib.load('../models/scaler.pkl')  # Load the scaler used in model training

# User input values (simulating what was previously handled by Streamlit)
credit_score = 400  # Example: Credit Score
gender = "Male"     # Example: Gender
age = 72            # Example: Age (raw value)
tenure = 2          # Example: Tenure in years
balance = 5000.0   # Example: Account balance (raw value)
products_number = 1 # Example: Number of products
credit_card = "Yes" # Example: Credit card ownership
active_member = "Yes" # Example: Active member
estimated_salary = 60000.0  # Example: Estimated salary
country = "France"  # Example: Country (could be France, Germany, or Spain)

# Process categorical inputs
gender = 1 if gender == "Male" else 0
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

# One-hot encoding for country
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0
# France is default, so no need to encode it (both country_Germany and country_Spain will be 0)

# Only scale the features that were scaled during training (balance, age, estimated_salary)
raw_input_data = np.array([[balance, age, estimated_salary]])  # Shape the raw data for scaling
scaled_features = scaler.transform(raw_input_data)[0]  # Apply scaling using the trained scaler

# Combine all features into a list (scaled features + other raw features)
input_data = [
    credit_score,      # Raw credit score
    gender,            # 1 for male, 0 for female
    scaled_features[1],  # Scaled age
    tenure,            # Raw tenure (not scaled)
    scaled_features[0],  # Scaled balance
    products_number,   # Raw number of products
    credit_card,       # 1 for yes, 0 for no (credit card)
    active_member,     # 1 for yes, 0 for no (active member)
    scaled_features[2],  # Scaled estimated salary
    country_Germany,   # One-hot encoded country (Germany)
    country_Spain      # One-hot encoded country (Spain)
]

# Flask API endpoint URL (change this as per your setup)
url = 'http://127.0.0.1:5000/predict'

# Send the processed data to the Flask API for prediction
response = requests.post(url, json={'features': input_data})

# Check the response from the API
if response.status_code == 200:
    prediction = response.json()
    if prediction['churn'] == 1:
        print(f"The customer is likely to churn. Probability: {prediction['churn_probability']:.2f}")
    else:
        print(f"The customer is not likely to churn. Probability: {prediction['churn_probability']:.2f}")
else:
    print("Failed to get prediction. Status Code:", response.status_code)
