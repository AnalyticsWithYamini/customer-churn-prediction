import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib

# Load the dataset (already cleaned for missing values)
file_path = '../data/ABC_Bank_Customer_Churn.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['customer_id'])

# Handle missing values (use Week 1 code if needed)
data['estimated_salary'].fillna(data['estimated_salary'].mean(), inplace=True)
data['gender'].fillna(data['gender'].mode()[0], inplace=True)

# Feature Engineering: Handling Outliers (Using IQR method for numeric columns)
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(), df[column])
    return df

# Handle outliers in balance, age, and estimated_salary
for col in ['balance', 'age', 'estimated_salary']:
    data = handle_outliers(data, col)

# Feature Scaling (Standardization)
scaler = StandardScaler()
data[['balance', 'age', 'estimated_salary']] = scaler.fit_transform(data[['balance', 'age', 'estimated_salary']])
joblib.dump(scaler, '../models/scaler.pkl')
# Encoding categorical columns (One-hot encoding for 'country', Label Encoding for 'gender')
data = pd.get_dummies(data, columns=['country'], drop_first=True)  # One-hot encode 'country'
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # Label encode 'gender'

# Save the preprocessed dataset
data.to_csv('../data/ABC_Bank_Customer_Churn_preprocessed.csv', index=False)

# Check processed data
print("Processed Data Sample:")
print(data.head())
