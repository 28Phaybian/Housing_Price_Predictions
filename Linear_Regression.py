import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'housing_price_dataset.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Preprocess the data
data = data.dropna()  # Drop rows with missing values
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Use the correct column names for Year and Price
year_column = "YearBuilt"
price_column = "Price"

# Ensure the dataset contains the required columns
if year_column not in data.columns or price_column not in data.columns:
    raise ValueError(f"Dataset must contain '{year_column}' and '{price_column}' columns!")

# Separate features and target
X = data[[year_column]]  # Feature: Year Built
y = data[price_column]   # Target: Price

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = linear_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression - Mean Squared Error: {mse}")
print(f"Linear Regression - R² Score: {r2}")

# Visualize Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted Prices")
# Add the red line (perfect prediction line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
plt.legend()
plt.grid()
plt.show()

# Forecast Future Prices (2024–2031)
future_years = pd.DataFrame({year_column: np.arange(2024, 2032)})  # Adjust the range to stop at 2031
future_predictions = linear_model.predict(future_years)

# Visualize Future Forecast
plt.figure(figsize=(10, 6))
plt.plot(data[year_column], y, label="Historical Prices", marker="o", linestyle="-")
plt.plot(future_years[year_column], future_predictions, label="Forecast Prices (2024–2031)", color="red", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Average Home Price")
plt.title("Linear Regression: Home Price Prediction (2024–2031)")
plt.legend()
plt.grid()
plt.show()

# Display Future Predictions
forecast_df = pd.DataFrame({
    "Year": future_years[year_column],
    "Predicted Price": future_predictions
})

print("Future Price Forecast (2024–2031):")
print(forecast_df)

# Calculate the percentage increase in home prices from 2024 to 2031
price_2024 = future_predictions[0]  # Price for 2024 (first entry)
price_2031 = future_predictions[-1]  # Price for 2031 (last entry)

# Calculate the percentage increase
percentage_increase = ((price_2031 - price_2024) / price_2024) * 100

print(f"Percentage increase in home price from 2024 to 2031: {percentage_increase:.2f}%")
