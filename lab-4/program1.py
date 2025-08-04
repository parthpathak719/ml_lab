import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load():
    return pd.read_excel("lab-4/dataset.xlsx", sheet_name="National_Health_Interview_Surve")

def preprocess(df):
    df=df[df["Data_Value"]!="Value suppressed"]
    df["Data_Value"]=pd.to_numeric(df["Data_Value"],errors="coerce")
    df=df.dropna(subset=["Data_Value"])

    features=df[["YearStart","YearEnd"]]
    labels=df["RiskFactor"]

    le=LabelEncoder()
    labels_encoded=le.fit_transform(labels)

    return features,labels_encoded,le

def train_model(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    return knn,X_train,X_test,y_train,y_test

def evaluate(knn,X_train,X_test,y_train,y_test,le):
    print("Training Evaluation:")
    train_preds=knn.predict(X_train)
    print("Confusion Matrix (Train):")
    print(confusion_matrix(y_train,train_preds))
    print("Classification Report (Train):")
    print(classification_report(y_train,train_preds,target_names=le.classes_,zero_division=0))

    print("\nTest Evaluation:")
    test_preds=knn.predict(X_test)
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test,test_preds))
    print("Classification Report (Test):")
    print(classification_report(y_test,test_preds,target_names=le.classes_,zero_division=0))

def main():
    df=load()
    X,y,le=preprocess(df)
    knn,X_train,X_test,y_train,y_test=train_model(X,y)
    evaluate(knn,X_train,X_test,y_train,y_test,le)

main()
