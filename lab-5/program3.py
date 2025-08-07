import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean_data(filepath, sheet_name, features, target="Data_Value"):
    # Load dataset
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    # Filter out suppressed values
    df = df[df[target] != "Value suppressed"]
    # Convert target and features to numeric
    df[target] = pd.to_numeric(df[target], errors="coerce")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + [target])
    return df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def print_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(dataset_name + " Metrics:")
    print(" MSE  :", round(mse, 4))
    print(" RMSE :", round(rmse, 4))
    print(" MAPE :", round(mape, 2), "%")
    print(" R2   :", round(r2, 4))
    print()


def plot_all_features_together(results, features):
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(features):
        # Each 'results' entry is a tuple: (X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)
        X_train, y_train, y_train_pred, X_test, y_test, y_test_pred = results[feature]

        plt.subplot(len(features), 2, 2*i+1)
        plt.scatter(X_train, y_train, color='blue', s=10, alpha=0.6, label='Actual')
        plt.scatter(X_train, y_train_pred, color='red', marker='x', s=20, alpha=0.7, label='Predicted')
        if i == 0:
            plt.legend()
        plt.xlabel(feature)
        plt.ylabel('Data_Value')
        plt.title(f'Train: {feature}')

        plt.subplot(len(features), 2, 2*i+2)
        plt.scatter(X_test, y_test, color='blue', s=10, alpha=0.6, label='Actual')
        plt.scatter(X_test, y_test_pred, color='red', marker='x', s=20, alpha=0.7, label='Predicted')
        if i == 0:
            plt.legend()
        plt.xlabel(feature)
        plt.ylabel('Data_Value')
        plt.title(f'Test: {feature}')
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)  # Increase as needed for more space between rows
    plt.show()



def run_regression_per_feature_collect(df, features, target="Data_Value"):
    results = {}  # will store data for plotting later
    for feature in features:
        X = df[[feature]].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(X_train, y_train)
        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)

        print("Feature:", feature)
        print_metrics(y_train, y_train_pred, "Training Set")
        print_metrics(y_test, y_test_pred, "Test Set")
        print("-" * 50)

        # Collect results for plotting
        results[feature] = (X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)
    return results


if __name__ == "__main__":
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    features_to_use = ["YearStart", "YearEnd", "Sample_Size", "LocationID"]

    data = load_and_clean_data(filepath, sheet_name, features_to_use)
    results = run_regression_per_feature_collect(data, features_to_use)
    plot_all_features_together(results, features_to_use)
