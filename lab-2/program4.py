import pandas as pd
import numpy as np

def load():
 return pd.read_excel("lab-2/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def missing(df):
 return df.isna().sum()

def mean_var(df):
 n=df.select_dtypes(include=['int64','float64'])
 return n.mean(),n.var()

def outliers(df):
 n=df.select_dtypes(include=['int64','float64'])
 return ((n<=(n.mean()-3*n.std()))|(n>=(n.mean()+3*n.std()))).sum()

def encoding_type(df):
 c=df.select_dtypes(include=['object'])
 e={}
 for col in c.columns:
  u=c[col].unique()
  if len(u)==2:
   e[col]="Label"
  else:
   e[col]="OneHot"
 return e

df=load()
m=missing(df)
mean,var=mean_var(df)
o=outliers(df)
e=encoding_type(df)

print(m.to_string())
print("\n",mean.to_string())
print("\n",var.to_string())
print("\n",o.to_string(),"\n")
for k,v in e.items():
 print(k+":",v)