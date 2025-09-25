import numpy as np
import matplotlib.pyplot as plt

# AND gate dataset
def load_and_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[0],[0],[1]])  # AND truth table
    return X, y

# Sigmoid + derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network training (1 hidden layer, backpropagation)
def train_nn(X, y, hidden_neurons=2, lr=0.05, epochs=1000, tol=0.002):
    np.random.seed(42)
    n_samples, n_features = X.shape
    n_outputs = y.shape[1]

    # Initialize weights
    W1 = np.random.randn(n_features, hidden_neurons)   # input → hidden
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, n_outputs)    # hidden → output
    b2 = np.zeros((1, n_outputs))

    errors = []

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Compute error (MSE)
        error = y - a2
        sum_sq_error = np.mean(error**2)
        errors.append(sum_sq_error)

        # Backpropagation
        d_a2 = error * sigmoid_derivative(a2)
        d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)

        # Weight updates
        W2 += lr * np.dot(a1.T, d_a2)
        b2 += lr * np.sum(d_a2, axis=0, keepdims=True)
        W1 += lr * np.dot(X.T, d_a1)
        b1 += lr * np.sum(d_a1, axis=0, keepdims=True)

        if sum_sq_error <= tol:
            break

    return W1, b1, W2, b2, errors, epoch+1

# Prediction
def predict(X, W1, b1, W2, b2):
    a1 = sigmoid(np.dot(X, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    return (a2 >= 0.5).astype(int)

# Main
X, y = load_and_data()
W1, b1, W2, b2, errors, epochs_run = train_nn(X, y)

print("Epochs to Converge:", epochs_run)
print("Final Error:", errors[-1])

# Predictions
y_pred = predict(X, W1, b1, W2, b2)
print("Predictions:", y_pred.ravel())
print("True Labels:", y.ravel().tolist())

# Plot error vs epochs
plt.plot(errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Error vs Epochs (AND Gate Neural Network, Backpropagation)")
plt.grid(True)
plt.show()
