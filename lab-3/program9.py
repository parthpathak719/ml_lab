import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_and_preprocess(filepath, sheet_name, drop_columns):
    """Load dataset, drop irrelevant columns, encode categoricals, fill NaNs, and filter two classes."""
    data = pd.read_excel(filepath, sheet_name=sheet_name)

    # Drop irrelevant columns
    data = data.drop(columns=drop_columns, errors='ignore')

    # Keep non-null Category rows
    data = data[data['Category'].notna()]

    # Filter for first two distinct categories
    two_classes = data['Category'].unique()[:2]
    data = data[data['Category'].isin(two_classes)].copy()

    # Encode categorical columns except Category
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Category']

    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Fill numeric NaNs with mean
    for col in data.select_dtypes(include=['number']).columns:
        data[col] = data[col].fillna(data[col].mean())

    # Drop any remaining NaNs
    data = data.dropna()

    # Prepare features and labels
    X = data.drop(columns=['Category'])
    y = LabelEncoder().fit_transform(data['Category'])

    return X, y


def train_evaluate_knn(X_train, y_train, X_test, y_test, k_values):
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    return train_accuracies, test_accuracies


def plot_accuracies(k_values, train_acc, test_acc):
    plt.plot(k_values, train_acc, label="Train Accuracy", marker='o')
    plt.plot(k_values, test_acc, label="Test Accuracy", marker='s')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("kNN Accuracy vs k")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_confusion_and_report(knn, X, y, dataset_name="Dataset"):
    y_pred = knn.predict(X)
    print(f"\nConfusion Matrix - {dataset_name}:\n", confusion_matrix(y, y_pred))
    print(f"\nClassification Report - {dataset_name}:\n", classification_report(y, y_pred))


def infer_learning(train_acc, test_acc):
    if train_acc < 0.8 and test_acc < 0.8:
        print("\nInference: Model is UNDERFITTING.")
    elif train_acc > 0.95 and (train_acc - test_acc) > 0.1:
        print("\nInference: Model is OVERFITTING.")
    else:
        print("\nInference: Model has a REGULAR FIT.")


def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    drop_cols = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit','Data_Value_Type', 'Data_Value_Footnote_Symbol',
        'Data_Value_Footnote', 'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation', 'Geographic Level', 'StateAbbreviation'
    ]

    # Load and preprocess data
    X, y = load_and_preprocess(filepath, sheet_name, drop_cols)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    k_values = list(range(1, 12, 2))  # k = 1, 3, ..., 11
    train_acc, test_acc = train_evaluate_knn(X_train, y_train, X_test, y_test, k_values)

    # Plot accuracy vs k
    plot_accuracies(k_values, train_acc, test_acc)

    # Best k based on test accuracy
    best_index = test_acc.index(max(test_acc))
    best_k = k_values[best_index]
    print(f"\nBest k based on test accuracy: {best_k}")

    # Train best model and get reports
    best_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    print_confusion_and_report(best_knn, X_train, y_train, "Train")
    print_confusion_and_report(best_knn, X_test, y_test, "Test")

    # Infer model fit type
    infer_learning(train_acc[best_index], test_acc[best_index])


if __name__ == "__main__":
    main()
