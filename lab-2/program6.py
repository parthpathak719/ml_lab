import pandas as pd
import numpy as np

def load():
 return pd.read_excel("lab-2/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def cosine(a,b):
 return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

df=load()
df=df.applymap(lambda x:1 if x=='t' else 0 if x=='f' else x)
df=df.select_dtypes(include=['int64','float64'])
v1=df.iloc[0]
v2=df.iloc[1]
print("Cosine Similarity:",cosine(v1,v2))
