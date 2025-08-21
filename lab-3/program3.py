import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import LabelEncoder


def load_and_drop_columns(filepath, sheet_name, drop_columns):
    """Load dataset and drop irrelevant columns."""
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data = data.drop(columns=drop_columns, errors='ignore')
    return data


def preprocess_data(data):
    """Preprocess data: encode categorical features and fill NaNs in numeric features."""
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Category']
    
    label_encoders = {}
    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].mean())

    return data, label_encoders


def get_feature_vectors(data):
    """Drop Category column and get first two feature vectors as float arrays."""
    features_only = data.drop(columns=['Category'], errors='ignore')
    vec1 = features_only.iloc[0].values.astype(float)
    vec2 = features_only.iloc[1].values.astype(float)
    return vec1, vec2


def compute_minkowski_distances(vec1, vec2, r_values):
    """Calculate Minkowski distances between two vectors for given r values."""
    return [minkowski(vec1, vec2, p=r) for r in r_values]


def plot_distances(r_values, distances):
    """Plot Minkowski distances versus r."""
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, distances, marker='o', color='purple')
    plt.xlabel("r value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs r")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def print_distances(r_values, distances):
    """Print Minkowski distances for each r value."""
    print("Minkowski Distances between Row 0 and Row 1:")
    for r, dist in zip(r_values, distances):
        print(f"r = {r}: Distance = {dist:.4f}")


def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    drop_columns = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
        'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation',
        'Geographic Level', 'StateAbbreviation'
    ]

    data = load_and_drop_columns(filepath, sheet_name, drop_columns)
    data, label_encoders = preprocess_data(data)
    vec1, vec2 = get_feature_vectors(data)

    r_values = range(1, 11)
    distances = compute_minkowski_distances(vec1, vec2, r_values)

    plot_distances(r_values, distances)
    print_distances(r_values, distances)


if __name__ == "__main__":
    main()
