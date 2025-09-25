import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset and prepare AND gate data
def load_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")

    # Take first 4 rows and convert two columns into binary features
    sub = df.head(4).copy()
    sub['A'] = (sub['Sample_Size'] > sub['Sample_Size'].median()).astype(int)
    sub['B'] = (sub['LocationID'] % 2).astype(int)

    # AND gate target
    sub['Target'] = sub['A'] & sub['B']

    X = sub[['A', 'B']].values
    y = sub['Target'].values
    return X, y

# Step activation
def step_activation(x):
    return 1 if x > 0 else 0

# Perceptron training
def perceptron_train(X, y, lr=0.05, epochs=1000, tol=0.002):
    # Initial weights
    w0, w1, w2 = 10, 0.2, -0.75
    weights = np.array([w0, w1, w2], dtype=float)
    errors = []

    # Add bias input
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

    for epoch in range(epochs):
        sum_sq_error = 0
        for xi, target in zip(X_aug, y):
            net_input = np.dot(weights, xi)
            output = step_activation(net_input)
            error = target - output
            weights += lr * error * xi
            sum_sq_error += error**2
        errors.append(sum_sq_error)

        if sum_sq_error <= tol:
            break

    return weights, errors, epoch + 1

# Run the experiment
X, y = load_data()
weights, error_list, epochs_run = perceptron_train(X, y)

# Print results
print("Final Weights:", weights)
print("Epochs to Converge:", epochs_run)
print("Last Error:", error_list[-1])

# Plot error vs epochs
plt.plot(range(1, len(error_list)+1), error_list, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Error Convergence for AND Gate Perceptron")
plt.grid()
plt.show()


