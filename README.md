# Exam-Problem# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# File path
file_path = r'C:\Users\PerryK08\Downloads\Restaurant Revenue.xlsx'

# Load the data from the Excel file
restaurant_data = pd.read_excel(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(restaurant_data.head())

# Extracting features and target variable
features = ['Number_of_Customers', 'Menu_Price', 'Average_Customer_Spending', 'Promotions', 'Reviews']
target = 'Monthly_Revenue'

X = restaurant_data[features]
y = restaurant_data[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Function to format currency
def format_currency(amount):
    return '${:,.2f}'.format(amount)

# Displaying predicted monthly revenue for the first few test examples
print("\nPredicted Monthly Revenue for the first few test examples:")
for i in range(5):
    print(f"Example {i+1}: {format_currency(y_pred[i])}")

# Note: Since the model is trained on scaled features, the predictions are in scaled form.
# You may need to inverse scale the predictions to get them back to the original scale.
