import pandas as pd
import numpy as np

def load_and_clean_data(filepath, sheet_name):
    """Load the dataset, clean suppressed values, and convert to numeric."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df[df["Data_Value"] != "Value suppressed"]
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
    df = df.dropna(subset=["Data_Value"])
    return df

def equal_width_binning(series, bins=4):
    """Perform equal-width binning and return bin labels."""
    return pd.cut(series, bins=bins, labels=False, include_lowest=True)

def calculate_entropy(labels):
    """Calculate entropy for categorical labels."""
    counts = labels.value_counts(normalize=True)  # probabilities
    return -np.sum(counts * np.log2(counts))

def main():
    # Step 1: Load and clean
    df = load_and_clean_data("dataset.xlsx", "National_Health_Interview_Surve")
    
    # Step 2: Equal-width binning
    df["Bin"] = equal_width_binning(df["Data_Value"], bins=4)
    
    # Step 3: Calculate entropy
    entropy_value = calculate_entropy(df["Bin"])
    print("Entropy:", entropy_value)

if __name__ == "__main__":
    main()
