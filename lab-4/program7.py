import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def create_training_data():
    X=[]
    labels=[]
    for i in range(20):
        x=random.uniform(1,10)
        y=random.uniform(1,10)
        label=random.randint(0,1)
        X.append([x,y])
        labels.append(label)
    return np.array(X),np.array(labels)

def create_test_data():
    test=[]
    for x in np.arange(0,10.1,0.1):
        for y in np.arange(0,10.1,0.1):
            test.append([x,y])
    return np.array(test)

def find_best_k(X,labels):
    values={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
    knn=KNeighborsClassifier()
    search=GridSearchCV(knn,values,cv=5)
    search.fit(X,labels)
    return search.best_estimator_

def get_colors(predicted):
    colors=[]
    for i in predicted:
        if(i==0):
            colors.append('blue')
        else:
            colors.append('red')
    return colors

def draw(X,colors,k):
    plt.scatter(X[:,0],X[:,1],c=colors,s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best k='+str(k))
    plt.show()

X_train,labels=create_training_data()
X_test=create_test_data()
model=find_best_k(X_train,labels)
output=model.predict(X_test)
colors=get_colors(output)
draw(X_test,colors,model.n_neighbors)
