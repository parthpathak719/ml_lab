import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# -------------------
# Load dataset
# -------------------
def load_data(file_path, drop_columns):
    data = pd.read_excel(file_path)
    df = data.drop(columns=drop_columns, errors='ignore')
    return df


# -------------------
# Preprocess dataset
# -------------------
def preprocess_data(df, target_col='Category', restrict_two_classes=True):
    # Keep only rows with non-null target
    df = df[df[target_col].notna()]

    # Optionally restrict to 2 classes
    if restrict_two_classes:
        unique_categories = df[target_col].unique()
        if len(unique_categories) >= 2:
            class1_name, class2_name = unique_categories[:2]
            df = df[df[target_col].isin([class1_name, class2_name])]

    # Select categorical columns except target
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]

    # Fill missing categorical values
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Drop remaining NaNs
    df = df.dropna()

    # Features and labels
    X = df.drop(columns=[target_col])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)  # Ensure numeric
    X = np.ascontiguousarray(X.values)  # FIX: make C-contiguous NumPy array

    y = LabelEncoder().fit_transform(df[target_col])

    return X, y, label_encoders


# -------------------
# Train KNN
# -------------------
def train_knn(X, y, n_neighbors=3, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    return knn, X_train, X_test, y_train, y_test


# -------------------
# Evaluate Model
# -------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Confusion Matrix - Train:\n", confusion_matrix(y_train, y_train_pred))
    print("\nConfusion Matrix - Test:\n", confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report - Train:")
    print(classification_report(y_train, y_train_pred))

    print("Classification Report - Test:")
    print(classification_report(y_test, y_test_pred))

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Fit diagnosis
    if train_acc < 0.8 and test_acc < 0.8:
        print("\nInference: Model is UNDERFITTING.")
    elif train_acc > 0.95 and (train_acc - test_acc) > 0.1:
        print("\nInference: Model is OVERFITTING.")
    else:
        print("\nInference: Model has a REGULAR FIT.")


# -------------------
# Main workflow
# -------------------
def main():
    drop_columns = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
        'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Numerator', 'LocationID',
        'DataValueTyAutomated Detection and Counting of Windows using UAV Imagery based Remote SensingpeID',
        'GeoLocation', 'Geographic Level', 'StateAbbreviation'
    ]

    df = load_data('dataset.xlsx', drop_columns)
    X, y, encoders = preprocess_data(df, target_col='Category', restrict_two_classes=True)

    knn, X_train, X_test, y_train, y_test = train_knn(X, y, n_neighbors=3)
    evaluate_model(knn, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
