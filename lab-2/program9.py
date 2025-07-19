import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load():
 return pd.read_excel("lab-2/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def normalize(df):
 num=df.select_dtypes(include='number')
 mm=MinMaxScaler().fit_transform(num)
 zs=StandardScaler().fit_transform(num)
 return pd.DataFrame(mm,columns=num.columns),pd.DataFrame(zs,columns=num.columns)

df=load()
minmax,zscore=normalize(df)
print("Min-Max normalized:\n",minmax)
print("\n Z-score normalized:\n",zscore)
