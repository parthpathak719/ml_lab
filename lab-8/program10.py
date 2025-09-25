import numpy as np

def step_activation(x):
    return 1 if x >= 0 else 0

def train_perceptron_multioutput(X, Y, learning_rate=0.05, epochs=1000, error_threshold=0.002):
    np.random.seed(42)
    weights = np.random.uniform(-1, 1, (Y.shape[1], X.shape[1] + 1))  # rows=outputs

    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            xi = np.insert(X[i], 0, 1)  
            outputs = []
            for o in range(Y.shape[1]):
                net = np.dot(weights[o], xi)
                outputs.append(step_activation(net))
                error = Y[i, o] - outputs[o]
                weights[o] += learning_rate * error * xi
                total_error += error ** 2
        errors.append(total_error)
        if total_error <= error_threshold:
            break
    return weights, errors, epoch + 1

# AND Gate example with 2-output encoding
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([[1,0],[1,0],[1,0],[0,1]])  # encoded

weights, errors, epochs = train_perceptron_multioutput(X_and, y_and)

print("Final Weights:\n", weights)
print("Converged in epochs:", epochs)
