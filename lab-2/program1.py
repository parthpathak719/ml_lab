import numpy as np

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
    C=np.array([
        [386],
        [289],
        [393],
        [110],
        [280],
        [167],
        [271],
        [274],
        [148],
        [198]
    ])
    return A,C

def cost(A,C):
    return np.linalg.matrix_rank(A),np.linalg.pinv(A) @ C

A,C=create()
rank,X=cost(A,C)
print("Rank of A:",rank)
print("Cost per product (X):\n",X)