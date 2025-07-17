def createA(a1,a2):
    A=[]
    for i in range(a1):
        row1=[]
        for j in range(a2):
            a=int(input("Enter element for matrix A:"))
            row1.append(a)
        A.append(row1)
    return A

def createB(b1,b2):
    B=[]
    for i in range(b1):
        row2=[]
        for j in range(b2):
            b=int(input("Enter element for matrix B:"))
            row2.append(b)
        B.append(row2)
    return B

def product(A,B,a1,a2,b1,b2):
    if(a2==b1):
        C=[]
        for i in range(a1):
            row3=[]
            for j in range(b2):
                sum_product=0
                for k in range(a2):
                    sum_product+=(A[i][k]*B[k][j])
                row3.append(sum_product)
            C.append(row3)
        return C
    else:
        return -1

a1=int(input("Enter number of rows of matrix A:"))
a2=int(input("Enter number of columns of matrix A:"))
b1=int(input("Enter number of rows of matrix B:"))
b2=int(input("Enter number of columns of matrix B:"))
A=createA(a1,a2)
B=createB(b1,b2)
print("Matrix A:",A)
print("Matrix B:",B)
C=product(A,B,a1,a2,b1,b2)
if(C==-1):
    print("A and B are not multipliable")
else:
    print("Matrix AB:",C)