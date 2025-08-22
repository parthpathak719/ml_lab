import pandas as pd
import numpy as np

def load():
 return pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def impute(df):
 for col in df.columns:
  if df[col].isnull().sum()>0:
   if df[col].dtype=='object':
    df[col]=df[col].fillna(df[col].mode()[0])
   else:
    if np.abs(df[col]-df[col].mean()).max()>3*df[col].std():
     df[col]=df[col].fillna(df[col].median())
    else:
     df[col]=df[col].fillna(df[col].mean())
 return df

df=load()
df=impute(df)
print(df)
