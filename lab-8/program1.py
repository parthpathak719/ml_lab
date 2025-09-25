import pandas as pd
import numpy as np

def load_data():
    # Make sure the sheet name matches exactly as in your Excel file
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
    return df

def summation_unit(x):
    return sum(x)

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def bipolar_step(x):
    if x >= 0:
        return 1
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x, alpha=0.01):
    if x >= 0:
        return x
    else:
        return alpha * x

def error_comparator(target, output):
    return target - output

df = load_data()
if "Sample_Size" not in df.columns:
    raise KeyError("Column 'Sample_Size' not found in the dataset.")
x = df["Sample_Size"].head(5).tolist()
x = df["Sample_Size"].head(5).tolist()

# Summation
s = summation_unit(x)
print("Summation:", s)

# Activation examples (for first value in x)
val = x[0]
print("Step:", step(val))
print("Bipolar Step:", bipolar_step(val))
print("Sigmoid:", sigmoid(val))
print("TanH:", tanh(val))
print("ReLU:", relu(val))
print("Leaky ReLU:", leaky_relu(val))

# Error comparator example
target = 100
output = val
print("Error:", error_comparator(target, output))
