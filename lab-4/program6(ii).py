import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def load():
    return pd.read_excel("lab-4/dataset.xlsx", sheet_name="National_Health_Interview_Surve")

def prepare_training_data(df):
    df = df.dropna(subset=["Data_Value", "YearStart", "RiskFactor"])
    df = df[df["RiskFactor"].isin(["Hypertension", "Smoking"])]
    df = df.sample(20, random_state=42)
    X_train = df[["YearStart", "Data_Value"]].values
    y_train = df["RiskFactor"].apply(lambda x: 0 if x == "Hypertension" else 1).values
    return X_train, y_train

def generate_test_data():
    x_range = np.arange(0, 10.1, 0.1)
    y_range = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    return test_points

def plot_test_predictions(test_points, y_pred):
    colors = ['blue' if label == 0 else 'red' for label in y_pred]
    plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=1)
    plt.title("kNN Classification of Test Data (k=3)")
    plt.xlabel("X feature")
    plt.ylabel("Y feature")
    plt.show()

df = load()
X_train, y_train = prepare_training_data(df)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
test_points = generate_test_data()
y_pred = knn.predict(test_points)
plot_test_predictions(test_points, y_pred)
