import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

# Clean: remove rows where Data_Value is 'Value suppressed' or NaN
df = df[df["Data_Value"] != "Value suppressed"]
df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value", "YearStart"])

# Choose ONE regression feature
X = df[["YearStart"]].values
y = df["Data_Value"].values

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
reg = LinearRegression().fit(X_train, y_train)

# Predict on training data
y_train_pred = reg.predict(X_train)

# Print regression details
print("Regression coefficient (slope):", reg.coef_[0])
print("Intercept:", reg.intercept_)
print("First 5 predicted values on training set:", y_train_pred[:5])

# Plot scatter plot: actual vs predicted on training data
plt.scatter(X_train, y_train, color='blue', label='Actual Data')
plt.scatter(X_train, y_train_pred, color='red', marker='x', label='Predicted Data')
plt.xlabel('YearStart')
plt.ylabel('Data_Value')
plt.title('Scatter plot: Actual vs Predicted on Training Data')
plt.legend()
plt.show()
