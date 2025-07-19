import numpy as np
from sklearn.tree import DecisionTreeClassifier

def create():
    A=np.array([
        [20,6,2],
        [16,3,6],
        [27,6,2],
        [19,1,2],
        [24,4,2],
        [22,1,5],
        [15,4,2],
        [18,4,2],
        [21,1,4],
        [16,2,4]
    ])
    payments=[386,289,393,110,280,167,271,274,148,198]
    y=np.array(['RICH' if p>200 else 'POOR' for p in payments])
    return A,y

X,y=create()
model=DecisionTreeClassifier()
model.fit(X,y)
result=model.predict(X)

for i in range(len(X)):
    print("Customer",i+1,"=",result[i])
