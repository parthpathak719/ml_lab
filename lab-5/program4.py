import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np

# Load and clean data
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
df = df[df["Data_Value"] != "Value suppressed"]
df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value", "YearStart"])

# Prepare feature (drop target 'Data_Value')
X = df[["YearStart"]].values  # Only using the feature for clustering

# Optional: Split into train/test (we only cluster on train)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Output
print("Cluster labels for training data:", labels[:10])
print("Cluster centers:\n", centers)

