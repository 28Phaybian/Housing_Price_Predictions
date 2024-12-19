import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

# Load the dataset
file_path = 'housing_price_dataset.csv'  # Update this with your dataset file path
data = pd.read_csv(file_path)

# Define the column to analyze for outliers
target_column = 'Price'

# Visualize outliers using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, y=target_column, color='skyblue')
plt.title(f"Boxplot of {target_column} (Outliers Highlighted)")
plt.ylabel(target_column)
plt.show()

# Highlight potential outliers
q1 = data[target_column].quantile(0.25)  # First quartile
q3 = data[target_column].quantile(0.75)  # Third quartile
iqr = q3 - q1  # Interquartile range
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[(data[target_column] < lower_bound) | (data[target_column] > upper_bound)]

# Visualize outliers in a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(data.index, data[target_column], alpha=0.6, label="Data")
plt.scatter(outliers.index, outliers[target_column], color='red', label="Outliers", alpha=0.8)
plt.axhline(y=lower_bound, color='blue', linestyle='--', label="Lower Bound")
plt.axhline(y=upper_bound, color='blue', linestyle='--', label="Upper Bound")
plt.title(f"Scatter Plot of {target_column} with Outliers Highlighted")
plt.xlabel("Index")
plt.ylabel(target_column)
plt.legend()
plt.show()

# Print outlier summary
print("Number of Outliers:", len(outliers))
print("Outliers:")
print(outliers)
