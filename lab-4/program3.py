import random
import matplotlib.pyplot as plt

def generate_data():
    X=[]
    Y=[]
    labels=[]
    for i in range(20):
        x=random.uniform(1,10)
        y=random.uniform(1,10)
        label=random.randint(0,1)
        X.append(x)
        Y.append(y)
        labels.append(label)
    return X,Y,labels

def get_colors(labels):
    colors=[]
    for i in labels:
        if(i==0):
            colors.append('blue')
        else:
            colors.append('red')
    return colors

def plot_data(X,Y,colors):
    plt.scatter(X,Y,c=colors)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Randomly Generated Data Points')
    plt.show()

X,Y,labels=generate_data()
colors=get_colors(labels)
plot_data(X,Y,colors)
