from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('../models/random_forest_churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON input
    features = np.array([data['features']])  # Convert to 2D array for model input
    prediction = model.predict(features)
    prediction_prob = model.predict_proba(features)[0][1]  # Probability of churn
    
    return jsonify({
        'churn': int(prediction[0]),        # 0 for no churn, 1 for churn
        'churn_probability': prediction_prob  # Probability of churn
    })

if __name__ == '__main__':
    # Set the host and port for Heroku and local development
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable set by Heroku, fallback to 5000 for local
    app.run(host='0.0.0.0', port=port)  # Make the server available on all interfaces
