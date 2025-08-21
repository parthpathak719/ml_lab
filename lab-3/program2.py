import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filepath, sheet_name):
    """Load dataset from Excel."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def get_numeric_feature_data(data, feature_name):
    """Extract the numeric feature data and drop NaN values."""
    return data[feature_name].dropna().values


def compute_histogram(feature_data, num_bins):
    """Compute histogram counts and bin edges using numpy."""
    return np.histogram(feature_data, bins=num_bins)


def print_histogram_data(feature_name, hist_counts, bin_edges):
    """Print histogram data bin counts and ranges."""
    print(f"Feature: {feature_name}")
    print("\n--- Histogram Data (numpy.histogram) ---")
    for i in range(len(hist_counts)):
        print(f"Bin {i+1}: {hist_counts[i]} values, Range: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")


def plot_histogram(feature_data, feature_name, num_bins):
    """Plot histogram using matplotlib."""
    plt.figure(figsize=(6, 4))
    plt.hist(feature_data, bins=num_bins, color='skyblue', edgecolor='black')
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def compute_statistics(feature_data):
    """Compute and return mean and variance of the feature data."""
    mean_val = np.mean(feature_data)
    variance_val = np.var(feature_data)
    return mean_val, variance_val


def print_statistics(mean_val, variance_val):
    """Print mean and variance values."""
    print("\n--- Statistics ---")
    print(f"Mean: {mean_val:.4f}")
    print(f"Variance: {variance_val:.4f}")


def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    feature_name = "Data_Value"
    num_bins = 10

    data = load_dataset(filepath, sheet_name)
    feature_data = get_numeric_feature_data(data, feature_name)

    hist_counts, bin_edges = compute_histogram(feature_data, num_bins)
    print_histogram_data(feature_name, hist_counts, bin_edges)

    plot_histogram(feature_data, feature_name, num_bins)

    mean_val, variance_val = compute_statistics(feature_data)
    print_statistics(mean_val, variance_val)


if __name__ == "__main__":
    main()
