import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load():
 return pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

def clean(df):
 return df.applymap(lambda x:1 if x=='t' else 0 if x=='f' else x).select_dtypes(include='number')

def jc(a,b):
 f11=np.sum((a==1)&(b==1))
 f10=np.sum((a==1)&(b==0))
 f01=np.sum((a==0)&(b==1))
 return f11/(f11+f10+f01)

def smc(a,b):
 f11=np.sum((a==1)&(b==1))
 f10=np.sum((a==1)&(b==0))
 f01=np.sum((a==0)&(b==1))
 f00=np.sum((a==0)&(b==0))
 return (f11+f00)/(f11+f10+f01+f00)

def cos(a,b):
 return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

df=load()
df=clean(df).iloc[:20]

jc_mat=np.zeros((20,20))
smc_mat=np.zeros((20,20))
cos_mat=np.zeros((20,20))

for i in range(20):
 for j in range(20):
  jc_mat[i][j]=jc(df.iloc[i],df.iloc[j])
  smc_mat[i][j]=smc(df.iloc[i],df.iloc[j])
  cos_mat[i][j]=cos(df.iloc[i],df.iloc[j])

sns.heatmap(jc_mat,annot=False)
plt.title("JC Heatmap")
plt.show()

sns.heatmap(smc_mat,annot=False)
plt.title("SMC Heatmap")
plt.show()

sns.heatmap(cos_mat,annot=False)
plt.title("Cosine Similarity Heatmap")
plt.show()
