import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve", nrows=100)
    # You can change 'RiskFactor' to any binary/multiclass label in your task
    y = LabelEncoder().fit_transform(df["RiskFactor"].astype(str))
    # Select simple feature columns; change/expand features as needed
    features = ["YearStart", "YearEnd", "Sample_Size", "LocationID"]
    X = df[features].fillna(0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_base_classifiers():
    # Choose or expand as preferred: DecisionTree, RandomForest, AdaBoost, SVC
    return [
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier()),
        ('svc', SVC(probability=True))
    ]

def get_metamodel():
    # Experiment here: e.g., LogisticRegression, DecisionTree, SVC, etc.
    return LogisticRegression()

def train_stacking_classifier(X_train, y_train, base_models, meta_model):
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model

def evaluate(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print("Stacking classifier accuracy:", score)

# Main run
X_train, X_test, y_train, y_test = load_data()
base_models = get_base_classifiers()
meta_model = get_metamodel()
model = train_stacking_classifier(X_train, y_train, base_models, meta_model)
evaluate(model, X_test, y_test)
