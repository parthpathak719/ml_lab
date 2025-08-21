import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess functions (reuse or redefine as needed)
def load_data(filepath, sheet_name):
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data = data.dropna(subset=["Age", "Sex", "Data_Value"])
    return data

def preprocess_features(data):
    X = data[["Age", "Sex"]]
    X = pd.get_dummies(X)
    return X

def preprocess_target(data):
    y = pd.cut(data["Data_Value"], bins=3, labels=[0, 1, 2])
    return y

def train_and_evaluate(X_train, y_train, X_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    
    data = load_data(filepath, sheet_name)
    X = preprocess_features(data)
    y = preprocess_target(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ks = list(range(1, 12, 2))  # k values: 1, 3, 5, ..., 11
    accuracies = []
    
    for k in ks:
        acc = train_and_evaluate(X_train, y_train, X_test, y_test, k)
        print(f"Accuracy for k={k}: {acc}")
        accuracies.append(acc)
    
    # Plot accuracy vs k
    plt.plot(ks, accuracies, marker='o')
    plt.title("kNN Classifier Accuracy vs k")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
