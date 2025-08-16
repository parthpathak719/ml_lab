import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------
# Load dataset
# -------------------
data = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

# -------------------
# Target variable
# -------------------
target_col = 'Category'
df = data.copy()

# Remove rows with missing target
df = df[df[target_col].notna()]

# -------------------
# Drop irrelevant columns
# -------------------
drop_columns = [
    'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
    'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
    'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation',
    'Geographic Level', 'StateAbbreviation'
]
df = df.drop(columns=drop_columns, errors='ignore')

# -------------------
# Custom binning function
# -------------------
def bin_column(series, bin_type="width", bins=4):
    """
    Converts continuous column to categorical bins.

    Parameters:
    - series: Pandas Series (numeric column)
    - bin_type: "width" (equal-width) or "frequency" (equal-frequency)
    - bins: number of bins

    Defaults: bin_type="width", bins=4
    """
    if not np.issubdtype(series.dtype, np.number):
        return series  # already categorical
    
    if bin_type == "width":
        return pd.cut(series, bins=bins, labels=False, duplicates='drop')
    elif bin_type == "frequency":
        return pd.qcut(series, q=bins, labels=False, duplicates='drop')
    else:
        raise ValueError("bin_type must be 'width' or 'frequency'")

# -------------------
# Separate categorical & numeric columns
# -------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from feature lists
if target_col in categorical_cols:
    categorical_cols.remove(target_col)
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# -------------------
# Fill NaNs
# -------------------
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -------------------
# Apply binning to numeric columns
# -------------------
for col in numeric_cols:
    df[col] = bin_column(df[col], bin_type="frequency", bins=4)  # change bin_type as needed

# -------------------
# Encode categorical features
# -------------------
for col in categorical_cols + numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target
target_le = LabelEncoder()
df[target_col] = target_le.fit_transform(df[target_col])

# -------------------
# Entropy
# -------------------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

# -------------------
# Information Gain
# -------------------
def information_gain(X_col, y):
    total_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (c / len(X_col)) * entropy(subset_y)
    return total_entropy - weighted_entropy

# -------------------
# Find best root feature
# -------------------
X = df.drop(columns=[target_col])
y = df[target_col]

best_feature = None
best_ig = -1
for col in X.columns:
    ig = information_gain(X[col], y)
    print(f"Feature: {col}, IG: {ig:.4f}")
    if ig > best_ig:
        best_feature = col
        best_ig = ig

print("\nBest Root Feature:", best_feature)
print("Max Information Gain:", best_ig)
