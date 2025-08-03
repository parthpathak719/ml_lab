import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data():
    X=[]
    labels=[]
    for i in range(20):
        x=random.uniform(1,10)
        y=random.uniform(1,10)
        label=random.randint(0,1)
        X.append([x,y])
        labels.append(label)
    return np.array(X),np.array(labels)

def generate_test_data():
    test=[]
    for x in np.arange(0,10.1,0.1):
        for y in np.arange(0,10.1,0.1):
            test.append([x,y])
    return np.array(test)

def classify_with_knn(X_train,labels,X_test):
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,labels)
    return knn.predict(X_test)

def get_colors(predictions):
    colors=[]
    for p in predictions:
        if(p==0):
            colors.append('blue')
        else:
            colors.append('red')
    return colors

def plot_test_data(X_test,colors):
    plt.scatter(X_test[:,0],X_test[:,1],c=colors,s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kNN Classification (k=1)')
    plt.show()

X_train,labels=generate_training_data()
X_test=generate_test_data()
predictions=classify_with_knn(X_train,labels,X_test)
colors=get_colors(predictions)
plot_test_data(X_test,colors)
