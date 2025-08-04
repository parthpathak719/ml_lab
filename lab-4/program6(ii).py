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

def plot_multiple_knn(X_train, y_train, test_points):
    k_values = [1, 3, 5, 7]
    plt.figure(figsize=(10, 10))

    for index in range(len(k_values)):
        k = k_values[index]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        predictions = model.predict(test_points)
        colors = ['blue' if p == 0 else 'red' for p in predictions]

        plt.subplot(2, 2, index+1)
        plt.scatter(test_points[:, 0], test_points[:, 1], c=colors, s=1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("kNN Classification (k = " + str(k) + ")")

    plt.tight_layout()
    plt.show()

df = load()
X_train, y_train = prepare_training_data(df)
test_points = generate_test_data()
plot_multiple_knn(X_train, y_train, test_points)
