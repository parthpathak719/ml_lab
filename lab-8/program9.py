import numpy as np

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Training function for perceptron
def train_perceptron(X, y, learning_rate=0.05, epochs=1000, error_threshold=0.002):
    # Initialize weights randomly
    np.random.seed(42)
    weights = np.random.uniform(-1, 1, X.shape[1] + 1)  # +1 for bias
    
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            xi = np.insert(X[i], 0, 1)  # add bias input
            net = np.dot(weights, xi)
            output = step_activation(net)
            error = y[i] - output
            weights += learning_rate * error * xi
            total_error += error ** 2
        errors.append(total_error)

        if total_error <= error_threshold:
            break
    return weights, errors, epoch + 1

# XOR Dataset
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

weights, errors, epochs = train_perceptron(X_xor, y_xor)

print("Final Weights:", weights)
print("Converged in epochs:", epochs)
print("Errors:", errors)
