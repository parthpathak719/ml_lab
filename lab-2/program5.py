import pandas as pd

def load():
 return pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def to_binary(df):
 b=df.applymap(lambda x:1 if x=='t' else 0 if x=='f' else x)
 return b

def binaryCols(df):
 return [c for c in df.columns if set(df[c].unique())=={'t','f'}]

def jaccardSmc(v1,v2):
 f11=((v1==1)&(v2==1)).sum()
 f00=((v1==0)&(v2==0)).sum()
 f10=((v1==1)&(v2==0)).sum()
 f01=((v1==0)&(v2==1)).sum()
 jc=f11/(f11+f10+f01)
 smc=(f11+f00)/(f11+f00+f10+f01)
 return jc,smc

df=load()
bcols=binaryCols(df)
bdf=to_binary(df[bcols])
v1=bdf.iloc[0]
v2=bdf.iloc[1]
jc,smc=jaccardSmc(v1,v2)

print("Jaccard Coefficient:",jc)
print("Simple Matching Coefficient:",smc)
