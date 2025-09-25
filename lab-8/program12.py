import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
file_path = "dataset.xlsx"   # <-- keep the file in your working directory
df = pd.read_excel(file_path, sheet_name="National_Health_Interview_Surve")

# Drop rows with missing RiskFactor or important features
df_clean = df.dropna(subset=["RiskFactor", "Sample_Size", "LocationID"])

# Select features and target
X = df_clean[["YearStart", "YearEnd", "Sample_Size", "LocationID"]]
y = df_clean["RiskFactor"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Define and train MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="logistic",
    solver="adam",
    learning_rate_init=0.05,
    max_iter=1000,
    random_state=42
)
mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

# Evaluate metrics
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

print("Accuracy:", acc)
print("Precision (Weighted):", precision)
print("Recall (Weighted):", recall)
print("F1-score (Weighted):", f1)
