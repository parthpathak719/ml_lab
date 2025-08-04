import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score

def load():
    return pd.read_excel("lab-4/Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")

def get_data():
    df=load()
    actual=df["Price"].tolist()
    predicted=df["Open"].tolist() 
    return actual,predicted

def calculate_metrics(actual,predicted):
    mse=mean_squared_error(actual,predicted)
    rmse=np.sqrt(mse)
    mape=mean_absolute_percentage_error(actual,predicted)*100
    r2=r2_score(actual,predicted)
    print("MSE:",mse)
    print("RMSE:",rmse)
    print("MAPE:",mape)
    print("R2 Score:",r2)

actual,predicted=get_data()
calculate_metrics(actual,predicted)
