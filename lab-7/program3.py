import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# 1. Load and preprocess your data
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
df = df[df["Data_Value"] != "Value suppressed"]
df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value", "RiskFactor"])  # Example: classification on RiskFactor

# Encode target as numbers
le = LabelEncoder()
df["RiskFactor"] = le.fit_transform(df["RiskFactor"])

# Simple feature selection: use some columns for features (modify as needed)
features = ["YearStart", "YearEnd", "Sample_Size", "LocationID"]
X = df[features].fillna(0)  # fill missing values if any
y = df["RiskFactor"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. List of classifiers
classifiers = {
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "CatBoost": CatBoostClassifier(silent=True),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=300)
}

# 3. Results Table Setup
results = []

# 4. Train, test and collect metrics
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    # Train set predictions
    y_train_pred = model.predict(X_train)
    # Test set predictions
    y_test_pred = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    train_prec = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)

    results.append([
        name, round(train_acc, 3), round(test_acc, 3),
        round(train_f1, 3), round(test_f1, 3),
        round(train_prec, 3), round(test_prec, 3),
        round(train_recall, 3), round(test_recall, 3)
    ])

# 5. Tabulate
cols = [
    "Classifier", "Train Accuracy", "Test Accuracy",
    "Train F1", "Test F1", "Train Precision", "Test Precision", "Train Recall", "Test Recall"
]
df_results = pd.DataFrame(results, columns=cols)
print(df_results)
