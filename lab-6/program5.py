import pandas as pd
import numpy as np

# -----------------------------
# Entropy calculation
# -----------------------------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

# -----------------------------
# Information Gain
# -----------------------------
def information_gain(X_col, y):
    total_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for val, count in zip(values, counts):
        subset_y = y[X_col == val]
        weighted_entropy += (count / len(y)) * entropy(subset_y)
    return total_entropy - weighted_entropy

# -----------------------------
# Binning Function (from A4)
# -----------------------------
def bin_column(series, bins=4, method='frequency'):
    if pd.api.types.is_numeric_dtype(series):
        if method == 'frequency':
            return pd.qcut(series, q=bins, duplicates='drop').astype(str)
        elif method == 'width':
            return pd.cut(series, bins=bins, duplicates='drop').astype(str)
        else:
            raise ValueError("Method must be 'frequency' or 'width'")
    else:
        return series.astype(str)

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_data(df, target_col, bins=4, method='frequency'):
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        df[col] = bin_column(df[col], bins=bins, method=method)
    return df

# -----------------------------
# Decision Tree Builder
# -----------------------------
def build_tree(df, target_col, depth=0):
    y = df[target_col].values

    # Stop if all samples are of same class
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]

    # Stop if no features left
    features = [col for col in df.columns if col != target_col]
    if len(features) == 0:
        # Return most common class
        return pd.Series(y).mode()[0]

    # Choose feature with highest IG
    ig_scores = {feature: information_gain(df[feature].values, y) for feature in features}
    best_feature = max(ig_scores, key=ig_scores.get)

    # Create a node for this feature
    tree = {best_feature: {}}

    # Branch for each value of the feature
    for val in np.unique(df[best_feature]):
        subset = df[df[best_feature] == val].drop(columns=[best_feature])
        tree[best_feature][val] = build_tree(subset, target_col, depth + 1)

    return tree

# -----------------------------
# Prediction Function
# -----------------------------
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree  # Leaf node
    feature = next(iter(tree))
    feature_value = sample.get(feature)
    if feature_value in tree[feature]:
        return predict(tree[feature][feature_value], sample)
    else:
        return None  # Unknown branch

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Load dataset
    data = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
    target_col = "Category"

    # Drop irrelevant columns
    drop_cols = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
        'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation',
        'Geographic Level', 'StateAbbreviation'
    ]
    data = data.drop(columns=drop_cols, errors='ignore')

    # Drop rows with missing target
    data = data.dropna(subset=[target_col])

    # Fill missing values in features
    for col in data.columns:
        if col != target_col:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna("Unknown")

    # Convert features to categorical via binning
    data_cat = preprocess_data(data, target_col, bins=4, method='frequency')

    # Build Decision Tree
    tree = build_tree(data_cat, target_col)
    print("Decision Tree:")
    print(tree)

    # Example prediction with first row
    sample_dict = data_cat.iloc[0].drop(target_col).to_dict()
    pred = predict(tree, sample_dict)
    print("\nPrediction for first sample:", pred)
