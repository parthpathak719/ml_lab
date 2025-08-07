import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def prepare_data():
    """
    Function to prepare data for clustering
    """
    # Load dataset
    df = pd.read_excel('dataset.xlsx', sheet_name='National_Health_Interview_Surve', nrows=2000)
    
    # Select numerical features for clustering (excluding target variable)
    clustering_features = ['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']
    
    # Create training data by removing missing values
    X_train = df[clustering_features].dropna()
    
    # Standardize the features for better clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled

def elbow_method(X_train, k_range):
    """
    Function to implement elbow method for optimal k determination
    """
    distortions = []  # List to store inertia values
    
    for k in k_range:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_train)
        
        # Store the inertia (distortion)
        distortions.append(kmeans.inertia_)
        print(f"k={k}: Inertia={kmeans.inertia_:.4f}")
    
    return distortions

def plot_elbow_curve(k_range, distortions):
    """
    Function to plot elbow curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distortions, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Add annotations for each point
    for i, (k, dist) in enumerate(zip(k_range, distortions)):
        plt.annotate(f'k={k}', (k, dist), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.show()
    
    # Calculate rate of decrease to help identify elbow
    rate_of_decrease = []
    for i in range(1, len(distortions)):
        rate = distortions[i-1] - distortions[i]
        rate_of_decrease.append(rate)
        print(f"Decrease from k={k_range[i-1]} to k={k_range[i]}: {rate:.4f}")
    
    # Find potential elbow point (where rate of decrease changes significantly)
    if len(rate_of_decrease) > 1:
        rate_changes = []
        for i in range(1, len(rate_of_decrease)):
            rate_change = rate_of_decrease[i-1] - rate_of_decrease[i]
            rate_changes.append(rate_change)
        
        # The elbow is typically where the rate of decrease starts to level off
        max_rate_change_idx = np.argmax(rate_changes)
        suggested_k = k_range[max_rate_change_idx + 2]  # +2 because of indexing
        print(f"\nSuggested optimal k based on elbow method: {suggested_k}")

def calculate_elbow_score(distortions):
    """
    Function to calculate elbow score using knee locator method
    """
    # Simple method to find elbow point
    # Calculate the line from first to last point
    n_points = len(distortions)
    x_line = np.linspace(0, n_points-1, n_points)
    y_line = np.linspace(distortions[0], distortions[-1], n_points)
    
    # Calculate distances from each point to the line
    distances = []
    for i in range(n_points):
        # Distance from point to line
        distance = abs((y_line[-1] - y_line[0]) * i - (x_line[-1] - x_line[0]) * distortions[i] + 
                    x_line[-1] * y_line[0] - y_line[-1] * x_line[0]) / \
                    np.sqrt((y_line[-1] - y_line[0])**2 + (x_line[-1] - x_line[0])**2)
        distances.append(distance)
    
    # The elbow point is where the distance is maximum
    elbow_idx = np.argmax(distances)
    return elbow_idx

# Main execution for A7
if __name__ == "__main__":
    # Prepare training data
    X_train = prepare_data()
    print(f"Training data prepared with shape: {X_train.shape}")
    
    # Define range of k values for elbow method (typically 2 to 20)
    k_range = range(2, 20)
    
    # Apply elbow method
    print("Applying Elbow Method...")
    distortions = elbow_method(X_train, k_range)
    
    # Plot elbow curve
    plot_elbow_curve(k_range, distortions)
    
    # Calculate elbow point using knee locator approach
    elbow_idx = calculate_elbow_score(distortions)
    optimal_k_elbow = k_range[elbow_idx]
    print(f"\nOptimal k using elbow point calculation: {optimal_k_elbow}")
    
    # Additional analysis: show the percentage of variance explained
    print("\nVariance explained analysis:")
    for i, (k, dist) in enumerate(zip(k_range, distortions)):
        if i == 0:
            variance_explained = 0
        else:
            variance_explained = ((distortions[0] - dist) / distortions[0]) * 100
        print(f"k={k}: {variance_explained:.2f}% variance explained")
