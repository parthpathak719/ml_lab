import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------
# Load and preprocess data
# ------------------------
def load_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve", nrows=100)

    # Encode target variable
    y = LabelEncoder().fit_transform(df["RiskFactor"].astype(str))

    # Select features (change as needed)
    features = ["YearStart", "YearEnd", "Sample_Size", "LocationID"]
    X = df[features].fillna(0)

    return train_test_split(X, y, test_size=0.2, random_state=42), features

# ------------------------
# Base and Meta models
# ------------------------
def get_base_classifiers():
    return [
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier()),
        ('svc', SVC(probability=True))
    ]

def get_metamodel():
    return LogisticRegression()

# ------------------------
# Train stacking classifier
# ------------------------
def train_stacking_classifier(X_train, y_train, base_models, meta_model):
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=False
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model

# ------------------------
# Evaluate model
# ------------------------
def evaluate(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print("Stacking classifier accuracy:", score)

# ------------------------
# LIME explanation (only 1 sample)
# ------------------------
def explain_with_lime_one(model, X_train, X_test, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    # Pick just the first test instance
    i = 0
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i],
        predict_fn=model.predict_proba
    )

    # Show explanation plot on console
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for Test Instance {i}")
    plt.tight_layout()
    plt.show()

    print(f"Displayed LIME explanation for test instance {i}")

# ------------------------
# Main Run
# ------------------------
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), features = load_data()
    base_models = get_base_classifiers()
    meta_model = get_metamodel()
    model = train_stacking_classifier(X_train, y_train, base_models, meta_model)
    evaluate(model, X_test, y_test)

    # Run LIME on just one test sample
    class_labels = [str(c) for c in np.unique(y_train)]
    explain_with_lime_one(model, X_train, X_test, features, class_labels)
