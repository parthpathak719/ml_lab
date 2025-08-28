
# A2. Hyperparameter tuning with RandomizedSearchCV

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

file_path = "dataset.xlsx"
data = pd.read_excel(file_path, sheet_name="National_Health_Interview_Surve")

# Target: Response (Yes/No)
df_class = data.dropna(subset=["Response"])
y = df_class["Response"]

# Encode target
y_enc = LabelEncoder().fit_transform(y)

# Features
features = ["YearStart", "YearEnd", "Age", "Sex", "RaceEthnicity", "Sample_Size"]
X = df_class[features].copy()

# Encode categorical features
for col in ["Age", "Sex", "RaceEthnicity"]:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Fill missing numeric values
X["Sample_Size"] = X["Sample_Size"].fillna(X["Sample_Size"].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

param_dist = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,            # number of random combinations to try
    cv=5,                 # 5-fold cross validation
    verbose=2,
    random_state=42,
    n_jobs=1             
)

random_search.fit(X_train, y_train)

print("\nBest Hyperparameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred))
