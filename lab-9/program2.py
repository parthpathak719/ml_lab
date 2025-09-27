import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer

# -------------------
# Load dataset
# -------------------
data = pd.read_excel("dataset.xlsx",nrows=100)

# Drop irrelevant columns
drop_columns = [
    'Category', 'LocationAbbr', 'LocationDesc', 'DataSource',
    'Data_Value_Unit', 'Data_Value_Type', 'Data_Value_Footnote_Symbol',
    'Data_Value_Footnote', 'Numerator', 'LocationID', 'DataValueTypeID',
    'GeoLocation', 'Geographic Level', 'StateAbbreviation'
]
X = data.drop(columns=drop_columns, errors='ignore')

# Target
y = data['Category']

# -------------------
# Separate categorical & numeric columns
# -------------------
categorical_cols = [
    "Topic", "Question", "Response", "Age", "Sex", "RaceEthnicity",
    "RiskFactor", "RiskFactorResponse", "TopicID", "CategoryID",
    "QuestionID", "ResponseID", "AgeID", "SexID", "RaceEthnicityID",
    "RiskFactorID", "RiskFactorResponseID"
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# -------------------
# Preprocessor: handle categorical + numeric separately
# -------------------
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_cols),
    ("numeric", numeric_pipeline, numeric_cols)
])

# -------------------
# Build Pipeline
# -------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=min(10,X.shape[1]))),
    ("classifier", RandomForestClassifier(random_state=42))
])

# -------------------
# Train/Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print("Cross-validation Accuracy:", np.mean(scores))
print("Test Accuracy:", pipeline.score(X_test, y_test))