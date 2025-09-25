import numpy as np
from sklearn.neural_network import MLPClassifier

# AND Gate
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

clf_and = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', 
                        solver='adam', learning_rate_init=0.05, max_iter=1000, random_state=42)
clf_and.fit(X_and, y_and)
print("AND Predictions:", clf_and.predict(X_and))

# XOR Gate
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

clf_xor = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', 
                        solver='adam', learning_rate_init=0.05, max_iter=1000, random_state=42)
clf_xor.fit(X_xor, y_xor)
print("XOR Predictions:", clf_xor.predict(X_xor))
