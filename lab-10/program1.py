import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load dataset from Excel file."""
    return pd.read_excel(filepath)

def select_numerical_features(df):
    """Select only numerical columns for correlation."""
    return df.select_dtypes(include=['int64', 'float64'])

def plot_correlation_heatmap(df):
    """Plot a heatmap for feature correlation."""
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def main():
    df = load_data("dataset.xlsx")
    num_features = select_numerical_features(df)
    plot_correlation_heatmap(num_features)

if __name__ == "__main__":
    main()
