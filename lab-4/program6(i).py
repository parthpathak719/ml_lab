import pandas as pd
import matplotlib.pyplot as plt
import random

def load():
    return pd.read_excel("lab-4/dataset.xlsx", sheet_name="National_Health_Interview_Surve")

def preprocess(df):
    df=df[df["Data_Value"]!="Value suppressed"]
    df["Data_Value"]=pd.to_numeric(df["Data_Value"],errors="coerce")
    df=df.dropna(subset=["Data_Value"])

    # Select only two specific RiskFactor categories for binary classification
    df=df[df["RiskFactor"].isin(["Hypertension","Smoking"])]

    return df

def sample_and_plot(df):
    sample_df=df.sample(n=20,random_state=42)

    X=sample_df["YearStart"].tolist()
    Y=sample_df["Data_Value"].tolist()
    labels=sample_df["RiskFactor"].tolist()

    colors=[]
    for label in labels:
        if(label=="Hypertension"):
            colors.append("blue")  # class 0
        else:
            colors.append("red")   # class 1

    plt.scatter(X,Y,c=colors)
    plt.xlabel("YearStart")
    plt.ylabel("Data_Value")
    plt.title("Scatter Plot of 20 Training Data Points")
    plt.show()

def main():
    df=load()
    df=preprocess(df)
    sample_and_plot(df)

main()
