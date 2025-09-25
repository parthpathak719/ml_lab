import pandas as pd
import numpy as np

def load_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
    # Use binary demo as before (first 4 rows)
    sub = df.head(4).copy()
    sub['A'] = sub['Sample_Size'] > sub['Sample_Size'].median()
    sub['B'] = sub['LocationID'] % 2
    sub['Target'] = sub['A'] & sub['B']
    X = sub[['A', 'B']].astype(int).values
    y = sub['Target'].astype(int).values
    return X, y

def bipolar_step(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def perceptron_train(X, y, activation, lr=0.05, epochs=1000, tol=0.002):
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    weights = [w0, w1, w2]
    errors = []

    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

    for epoch in range(epochs):
        sum_sq_error = 0
        for xi, target in zip(X_aug, y):
            net_input = np.dot(weights, xi)
            output = activation(net_input)
            if activation == sigmoid: # Special case for sigmoid
                output = int(output >= 0.5)
            error = target - output
            for j in range(len(weights)):
                weights[j] += lr * error * xi[j]
            sum_sq_error += error**2
        errors.append(sum_sq_error)
        if sum_sq_error <= tol:
            break
    return weights, errors, epoch + 1

X, y = load_data()

weights_bipolar, error_bipolar, epochs_bipolar = perceptron_train(X, y, bipolar_step)
weights_sigmoid, error_sigmoid, epochs_sigmoid = perceptron_train(X, y, sigmoid)
weights_relu, error_relu, epochs_relu = perceptron_train(X, y, relu)

print("Bipolar Step - Epochs to Converge:", epochs_bipolar)
print("Sigmoid - Epochs to Converge:", epochs_sigmoid)
print("ReLU - Epochs to Converge:", epochs_relu)
