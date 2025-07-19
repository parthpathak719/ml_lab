import pandas as pd
import statistics as st
import matplotlib.pyplot as plt

def load():
    return pd.read_excel("lab-2/Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")

def stats(p):
    return st.mean(p),st.variance(p)

def wednesday_mean(df):
    return st.mean(df[df['Day']=='Wed'].iloc[:,3])

def april_mean(df):
    return st.mean(df[pd.to_datetime(df['Date']).dt.month==4].iloc[:,3])

def prob_loss(c):
    return len(list(filter(lambda x:x<0,c)))/len(c)

def prob_profit_wed(df):
    w=df[df['Day']=='Wed'].iloc[:,8]
    return len(list(filter(lambda x:x>0,w)))/len(w)

def cond_prob(df):
    w=df[df['Day']=='Wed']
    return len(w[w.iloc[:,8]>0])/len(w)

def plot(df):
    plt.scatter(df['Day'],df.iloc[:,8])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

df=load()
mean,var=stats(df.iloc[:,3])
w_mean=wednesday_mean(df)
a_mean=april_mean(df)
p_loss=prob_loss(df.iloc[:,8])
p_wed=prob_profit_wed(df)
p_cond=cond_prob(df)

print("Mean:",mean)
print("Variance:",var)
print("Wednesday Mean:",w_mean)
print("April Mean:",a_mean)
print("P(Loss):",round(p_loss,2))
print("P(Profit on Wed):",round(p_wed,2))
print("P(Profit|Wed):",round(p_cond,2))

plot(df)