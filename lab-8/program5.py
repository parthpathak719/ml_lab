import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
def load_xor_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])  # XOR truth table
    return X, y

# Activation functions
def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x > 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

# Perceptron training
def perceptron_train(X, y, activation, lr=0.05, epochs=1000, tol=0.002):
    weights = np.array([10, 0.2, -0.75])  # [bias, w1, w2]
    errors = []
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias

    for epoch in range(epochs):
        sum_sq_error = 0
        for xi, target in zip(X_aug, y):
            net_input = np.dot(weights, xi)

            if activation == sigmoid:
                out = 1 if sigmoid(net_input) >= 0.5 else 0
            elif activation == bipolar_step:
                out = 1 if net_input > 0 else -1
                target = 1 if target == 1 else -1
            else:  # step or ReLU
                out = 1 if activation(net_input) > 0 else 0

            error = target - out
            weights += lr * error * xi
            sum_sq_error += error**2

        errors.append(sum_sq_error)
        if sum_sq_error <= tol:
            break

    return weights, errors, epoch + 1

# Main
X, y = load_xor_data()

# Train with different activations
weights_step, error_step, epochs_step = perceptron_train(X, y, step)
weights_bipolar, error_bipolar, epochs_bipolar = perceptron_train(X, y, bipolar_step)
weights_sigmoid, error_sigmoid, epochs_sigmoid = perceptron_train(X, y, sigmoid)
weights_relu, error_relu, epochs_relu = perceptron_train(X, y, relu)

# Print results
print("Step        - Epochs:", epochs_step, " Final Error:", error_step[-1])
print("Bipolar Step- Epochs:", epochs_bipolar, " Final Error:", error_bipolar[-1])
print("Sigmoid     - Epochs:", epochs_sigmoid, " Final Error:", error_sigmoid[-1])
print("ReLU        - Epochs:", epochs_relu, " Final Error:", error_relu[-1])

# Plot error curves
plt.plot(error_step, label="Step")
plt.plot(error_bipolar, label="Bipolar Step")
plt.plot(error_sigmoid, label="Sigmoid")
plt.plot(error_relu, label="ReLU")
plt.xlabel("Epoch")
plt.ylabel("Sum of Squared Errors")
plt.title("Error vs Epochs (XOR Gate Perceptron)")
plt.legend()
plt.grid(True)
plt.show()