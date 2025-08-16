import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
print(df.columns.tolist())

# ==== Your chosen columns ====
features = ['Age', 'RaceEthnicity']
target = 'RiskFactor'

# Encode categorical columns (if necessary)
for col in features + [target]:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df[features]
y = df[target]

# Train the decision tree
dtc = DecisionTreeClassifier()
dtc.fit(X, y)

# Visualize the tree
plt.figure(figsize=(10, 7))
plot_tree(
    dtc, 
    feature_names=features, 
    class_names=[str(i) for i in sorted(y.unique())], 
    filled=True
)
plt.title("Decision Tree Visualization for Lab06")
plt.tight_layout()
plt.show()
