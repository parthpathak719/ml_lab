import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load customer dataset
def load_customer_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
    # Manually construct the given customer table
    data = {
        "Candies": [20,16,27,19,24,22,15,18,21,16],
        "Mangoes": [6,3,6,1,4,1,4,4,1,2],
        "Milk":    [2,6,2,2,2,5,2,2,4,4],
        "Payment": [386,289,393,110,280,167,271,274,148,198],
        "HighTx":  [1,1,1,0,1,0,1,1,0,0]  # Yes=1, No=0
    }
    df = pd.DataFrame(data)
    X = df[["Candies", "Mangoes", "Milk", "Payment"]].values
    y = df["HighTx"].values
    return X, y

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron training with Sigmoid
def perceptron_train(X, y, lr=0.01, epochs=1000, tol=0.002):
    n_features = X.shape[1]
    weights = np.random.randn(n_features + 1) * 0.01  # bias + features
    errors = []
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias column

    for epoch in range(epochs):
        sum_sq_error = 0
        for xi, target in zip(X_aug, y):
            net_input = np.dot(weights, xi)
            output = sigmoid(net_input)
            error = target - output
            weights += lr * error * xi  # update
            sum_sq_error += error**2

        errors.append(sum_sq_error)
        if sum_sq_error <= tol:
            break

    return weights, errors, epoch + 1

# Prediction
def predict(X, weights):
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
    outputs = sigmoid(np.dot(X_aug, weights))
    return (outputs >= 0.5).astype(int)

# Main
X, y = load_customer_data()
weights, error_list, epochs_run = perceptron_train(X, y, lr=0.01)

print("Final Weights:", weights)
print("Epochs to Converge:", epochs_run)
print("Final Error:", error_list[-1])

# Predictions
y_pred = predict(X, weights)
print("Predictions:", y_pred)
print("True Labels:", y.tolist())

# Accuracy
accuracy = np.mean(y_pred == y)
print("Training Accuracy:", accuracy)

# Plot Error vs Epochs
plt.plot(error_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Sum of Squared Errors")
plt.title("Error vs Epochs (Customer Data, Sigmoid Perceptron)")
plt.grid(True)
plt.show()