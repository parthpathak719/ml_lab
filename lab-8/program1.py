import numpy as np

# a) Summation unit
def summation_unit(x):
    return sum(x)

# b) Activation units
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

# c) Comparator unit for error calculation
def error_comparator(target, output):
    return target - output

# Example usage (remove or comment out when integrating elsewhere)
if __name__ == "__main__":
    x = [1, -2, 3]
    print("Summation:", summation_unit(x))
    print("Step:", step(0.5))
    print("Bipolar Step:", bipolar_step(-0.2))
    print("Sigmoid:", sigmoid(1))
    print("TanH:", tanh(1))
    print("ReLU:", relu(-2))
    print("Leaky ReLU:", leaky_relu(-2))
    print("Error:", error_comparator(2, 0.5))
