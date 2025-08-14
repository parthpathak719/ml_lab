import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---------- Entropy ----------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# ---------- Information Gain ----------
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = sum(
        (count / len(df)) * entropy(df[df[feature] == val][target])
        for val, count in zip(values, counts)
    )
    return total_entropy - weighted_entropy

# ---------- Find Best Root Node ----------
def find_best_root_node(df, target):
    features = [col for col in df.columns if col != target]
    gains = {f: information_gain(df, f, target) for f in features}
    best_feature = max(gains, key=gains.get)
    return best_feature, gains

# ---------- Plot Decision Tree ----------
def plot_decision_tree(X, y, feature_names):
    feature_names = list(feature_names)
    class_names = sorted(y.unique().astype(str))  # Ensure string class labels

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,  # <== LIMIT THE TREE DEPTH
        min_samples_leaf=10,
        random_state=0
    )
    clf.fit(X, y)

    plt.figure(figsize=(24, 12))  # Larger plot
    plot_tree(clf,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              fontsize=10)
    plt.title("Decision Tree (Depth limited to 4)")
    plt.show()

# ---------- Main ----------
def main():
    # Load dataset
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

    target = "Data_Value"
    df = df[df[target] != "Value suppressed"]
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    # Bin target into categories
    df[target] = pd.cut(df[target], bins=3, labels=["low", "mid", "high"])

    # Bin only numeric features with at least 2 unique values
    for col in df.select_dtypes(include=np.number).columns:
        if col != target and df[col].nunique() > 1:
            df[col] = pd.cut(df[col], bins=3,
                             labels=[f"{col}_low", f"{col}_mid", f"{col}_high"])

    # Drop any columns with NaN after binning
    df = df.dropna(axis=1, how="any")

    # Find best root node
    best_feature, gains = find_best_root_node(df, target)
    print("Information Gain for each feature:")
    for feature, gain in gains.items():
        print(f"{feature}: {gain:.4f}")
    print(f"\nBest Root Node Feature: {best_feature}")

    # Prepare for decision tree
    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]
    plot_decision_tree(X, y, X.columns)

if __name__ == "__main__":
    main()
