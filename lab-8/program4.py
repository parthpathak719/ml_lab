import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_excel("dataset.xlsx", sheet_name="National_Health_Interview_Surve")
    sub = df.head(4).copy()
    sub['A'] = sub['Sample_Size'] > sub['Sample_Size'].median()
    sub['B'] = sub['LocationID'] % 2
    sub['Target'] = sub['A'] & sub['B']
    X = sub[['A', 'B']].astype(int).values
    y = sub['Target'].astype(int).values
    return X, y

def step_activation(x):
    if x > 0:
        return 1
    else:
        return 0

def perceptron_train(X, y, lr, epochs=1000, tol=0.002):
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    weights = [w0, w1, w2]
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

    for epoch in range(epochs):
        sum_sq_error = 0
        for xi, target in zip(X_aug, y):
            net_input = np.dot(weights, xi)
            output = step_activation(net_input)
            error = target - output
            for j in range(len(weights)):
                weights[j] += lr * error * xi[j]
            sum_sq_error += error**2
        if sum_sq_error <= tol:
            return epoch + 1
    return epochs

X, y = load_data()
learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs_list = []

for rate in learning_rates:
    epochs_needed = perceptron_train(X, y, lr=rate)
    epochs_list.append(epochs_needed)

plt.plot(learning_rates, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Convergence")
plt.title("Learning Rate vs. Epochs to Converge")
plt.grid(True)
plt.show()

print("Learning Rate:", learning_rates)
print("Epochs to Converge:", epochs_list)
