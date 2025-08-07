import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load and clean data
df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

df = df[df["Data_Value"] != "Value suppressed"]
df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value", "YearStart"])

# Prepare feature (exclude target)
X = df[["YearStart"]].values

# Split data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)

# Evaluation metrics
sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_score = davies_bouldin_score(X_train, kmeans.labels_)

# Print the results
print("Silhouette Score:", round(sil_score, 4))
print("Calinski-Harabasz (CH) Score:", round(ch_score, 4))
print("Davies-Bouldin (DB) Index:", round(db_score, 4))
