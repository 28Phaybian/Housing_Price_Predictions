import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'housing_price_dataset.csv'  # Ensure this file exists in the same directory
data = pd.read_csv(file_path)

# Preprocess the data
data = data.dropna()  # Drop rows with missing values
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target (using all relevant features)
X = data.drop('Price', axis=1)  # Drop the target column 'Price'
y = data['Price']              # Target: Price

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - Mean Squared Error: {mse}")
print(f"Random Forest - R² Score: {r2}")

# Visualize the results: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted Prices")

# Add the red line (perfect prediction line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.show()

# Future Forecast (Using Years 2024-2031 and other features)
# Define future data with assumptions (8 years from 2024 to 2031)
future_years = pd.DataFrame({
    "SquareFeet": [2000 + (i * 50) for i in range(8)],  # Increase by 50 sqft each year for 8 years
    "Bedrooms": [3 + (i // 3) for i in range(8)],  # Increase the number of bedrooms after every 3 years
    "Bathrooms": [2 + (i // 5) for i in range(8)],  # Increase bathrooms after every 5 years
    "Neighborhood": [0] * 8,  # Keep neighborhood constant or adjust based on location trends
    "YearBuilt": list(range(2024, 2032))  # Years for prediction from 2024 to 2031 (8 years)
})

# Predict future prices
future_predictions = random_forest_model.predict(future_years)

# Plot Future Forecast
plt.figure(figsize=(10, 6))
plt.plot(data['YearBuilt'], data['Price'], label="Historical Prices", marker="o", linestyle="-")
plt.plot(future_years['YearBuilt'], future_predictions, label="Forecast Prices", color="red", linestyle="--")
plt.xlabel("Year Built / Predicted")
plt.ylabel("Average Home Price")
plt.title("Random Forest: Future Home Price Prediction (2024–2031)")
plt.legend()
plt.grid()
plt.show()

# Display Future Predictions
forecast_df = pd.DataFrame({
    "Year": future_years['YearBuilt'],
    "Predicted Price": future_predictions
})
print("Future Price Forecast (2024–2031):")
print(forecast_df)

# Calculate the percentage increase in home prices from 2024 to 2031
price_2024 = future_predictions[0]  # Price for 2024
price_2031 = future_predictions[7]  # Price for 2031 (8th year in the future_years array)

# Calculate the percentage increase
percentage_increase = ((price_2031 - price_2024) / price_2024) * 100

print(f"Percentage increase in home price from 2024 to 2031: {percentage_increase:.2f}%")
