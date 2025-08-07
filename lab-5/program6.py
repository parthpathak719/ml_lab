import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def prepare_data():
    """
    Function to prepare data for clustering
    """
    # Load dataset
    df = pd.read_excel('dataset.xlsx', sheet_name='National_Health_Interview_Surve', nrows=2000)
    
    # Select numerical features for clustering (excluding target variable)
    # Focus on health-related numerical features
    clustering_features = ['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']
    
    # Create training data by removing missing values
    X_train = df[clustering_features].dropna()
    
    # Standardize the features for better clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled

def evaluate_kmeans_for_k_values(X_train, k_range):
    """
    Function to evaluate k-means for different k values
    """
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    for k in k_range:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_train)
        
        # Calculate evaluation metrics
        sil_score = silhouette_score(X_train, cluster_labels)
        ch_score = calinski_harabasz_score(X_train, cluster_labels)
        db_score = davies_bouldin_score(X_train, cluster_labels)
        
        silhouette_scores.append(sil_score)
        ch_scores.append(ch_score)
        db_scores.append(db_score)
        
        print(f"k={k}: Silhouette={sil_score:.4f}, CH Score={ch_score:.4f}, DB Score={db_score:.4f}")
    
    return silhouette_scores, ch_scores, db_scores

def plot_evaluation_metrics(k_range, sil_scores, ch_scores, db_scores):
    """
    Function to plot evaluation metrics against k values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Silhouette Score
    axes[0].plot(k_range, sil_scores, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs k')
    axes[0].grid(True)
    
    # Plot CH Score
    axes[1].plot(k_range, ch_scores, 'go-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Calinski-Harabasz Score')
    axes[1].set_title('CH Score vs k')
    axes[1].grid(True)
    
    # Plot DB Score
    axes[2].plot(k_range, db_scores, 'ro-')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('DB Score vs k')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Determine optimal k values
    optimal_k_sil = k_range[np.argmax(sil_scores)]
    optimal_k_ch = k_range[np.argmax(ch_scores)]
    optimal_k_db = k_range[np.argmin(db_scores)]
    
    print(f"\nOptimal k based on:")
    print(f"Silhouette Score: k = {optimal_k_sil}")
    print(f"CH Score: k = {optimal_k_ch}")
    print(f"DB Score: k = {optimal_k_db}")

# Main execution for A6
if __name__ == "__main__":
    # Prepare training data
    X_train = prepare_data()
    print(f"Training data prepared with shape: {X_train.shape}")
    
    # Define range of k values to test
    k_range = range(2, 11)
    
    # Evaluate k-means for different k values
    sil_scores, ch_scores, db_scores = evaluate_kmeans_for_k_values(X_train, k_range)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(k_range, sil_scores, ch_scores, db_scores)
