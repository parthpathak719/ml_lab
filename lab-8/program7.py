import pandas as pd
import numpy as np

# Load customer dataset
def load_customer_data():
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

# Predict using pseudo-inverse weights
def pseudo_inverse_train(X, y):
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias
    weights = np.linalg.pinv(X_aug) @ y
    return weights

def predict(X, weights):
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
    outputs = np.dot(X_aug, weights)
    return (outputs >= 0.5).astype(int)

# Main
X, y = load_customer_data()

# Train with pseudo-inverse
weights_pinv = pseudo_inverse_train(X, y)
y_pred_pinv = predict(X, weights_pinv)
accuracy_pinv = np.mean(y_pred_pinv == y)

print("Pseudo-Inverse Weights:", weights_pinv)
print("Predictions:", y_pred_pinv)
print("True Labels:", y.tolist())
print("Accuracy (Pseudo-Inverse):", accuracy_pinv)
