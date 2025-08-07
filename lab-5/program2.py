import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load and clean data
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
df = df[df["Data_Value"] != "Value suppressed"]
df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value", "YearStart"])

# Prepare feature and target
X = df[["YearStart"]].values
y = df["Data_Value"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
reg = LinearRegression().fit(X_train, y_train)

# Predict
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Calculate MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Print metrics function
def print_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    print(name + " Metrics:")
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("MAPE:", round(mape(y_true, y_pred), 2), "%")
    print("R2:", round(r2_score(y_true, y_pred), 4))
    print()

# Output metrics
print_metrics(y_train, y_train_pred, "Training Set")
print_metrics(y_test, y_test_pred, "Test Set")
