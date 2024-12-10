import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset from CSV
file_path = 'water_dataX.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path, encoding='latin1')

# Display dataset information
print("First few rows:")
print(data.head())

# Handle missing or invalid values
data.replace('NAN', np.nan, inplace=True)  # Replace 'NAN' string with actual NaN

# Replace NaN values with 0 or drop rows with NaN values
data.fillna(0, inplace=True)  # Replace NaN with 0 

# Convert columns to numeric (where applicable)
numeric_columns = ['Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)', 'B.O.D. (mg/l)',
                   'NITRATENAN N+ NITRITENANN (mg/l)', 'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Verify no non-numeric values remain
print("\nAfter converting to numeric types:")
print(data[numeric_columns].info())

# Drop unnecessary columns
columns_to_drop = ['STATION CODE', 'LOCATIONS', 'STATE', 'year']
data = data.drop(columns=columns_to_drop)

# Define target (WQI calculation)
data['WQI'] = (data['D.O. (mg/l)'] * 0.3 +
               data['PH'] * 0.2 +
               data['B.O.D. (mg/l)'] * 0.4 +
               data['NITRATENAN N+ NITRITENANN (mg/l)'] * 0.1).round(2)

# Drop rows where 'WQI' is NaN (if needed)
data.dropna(subset=['WQI'], inplace=True)

# Split features and target
X = data.drop(columns=['WQI'])
y = data['WQI']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Error Calculations
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Errors
print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-Squared (R2 Score): {r2:.2f}")

# Feature Importance
importance = model.feature_importances_
features = X.columns if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

# Visualization - Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Visualization - Error Distribution
error = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(error, kde=True, bins=30, color='blue')
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Visualization - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted WQI')
plt.xlabel('Actual WQI')
plt.ylabel('Predicted WQI')
plt.show()

# Correlation Matrix (Heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Save Results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': error})
results.to_csv('wqi_predictions.csv', index=False)
print("\nResults saved to 'wqi_predictions.csv'")
