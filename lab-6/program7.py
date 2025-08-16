import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

features = ['Age', 'RaceEthnicity']   # Make sure these are EXACT
target = 'RiskFactor'

for col in features + [target]:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df[features]
y = df[target]

dtc = DecisionTreeClassifier()
dtc.fit(X, y)

# Directly use string names for min/max (not via features[0]/features[1])
x_min, x_max = df['Age'].min() - 1, df['Age'].max() + 1
y_min, y_max = df['RaceEthnicity'].min() - 1, df['RaceEthnicity'].max() + 1

print("x_min:", x_min, "| x_max:", x_max)
print("y_min:", y_min, "| y_max:", y_max)

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x='Age', y='RaceEthnicity', hue=target, data=df, palette='Set1', edgecolor='k')
plt.xlabel('Age')
plt.ylabel('RaceEthnicity')
plt.title("Decision Boundary using Decision Tree (Lab06)")
plt.tight_layout()
plt.show()
