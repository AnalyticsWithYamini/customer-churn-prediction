import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../data/ABC_Bank_Customer_Churn.csv'
data = pd.read_csv(file_path)

# Drop unused column
data = data.drop(columns=['customer_id'])

# Display basic information
print("Data Info:")
print(data.info())
print("\nData Description:")
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Handle missing values (using mean for 'estimated_salary' and mode for 'gender')
data['estimated_salary'].fillna(data['estimated_salary'].mean(), inplace=True)
data['gender'].fillna(data['gender'].mode()[0], inplace=True)

# Exploratory Data Analysis (Visualizations)
plt.figure(figsize=(10, 6))
sns.countplot(x='churn', data=data)
plt.title('Churn Distribution')
#plt.savefig('../output/eda_visualizations/churn_distribution.png')
plt.show()

# Visualize correlation between numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
#plt.savefig('../output/eda_visualizations/correlation_matrix.png')
plt.show()

# Boxplot to check outliers in numeric columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['estimated_salary', 'balance', 'age']])
plt.title('Boxplot for Numeric Features')
#plt.savefig('../output/eda_visualizations/boxplot_numeric.png')
plt.show()
