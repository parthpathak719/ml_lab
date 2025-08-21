import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

def train_knn_classifier(X_train, y_train, k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh

def evaluate_accuracy(neigh, X_test, y_test):
    accuracy = neigh.score(X_test, y_test)
    print("Test accuracy:", accuracy)

def predict_samples(neigh, X_test):
    predictions = neigh.predict(X_test)
    print("Predictions for first 5 test samples:")
    for i in range(5):
        print(f"Sample {i+1}: Predicted class = {predictions[i]}")

def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    
    data = load_data(filepath, sheet_name)
    X = preprocess_features(data)
    y = preprocess_target(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = train_knn_classifier(X_train, y_train, k=3)
    
    evaluate_accuracy(knn_model, X_test, y_test)
    
    predict_samples(knn_model, X_test)

if __name__ == "__main__":
    main()
