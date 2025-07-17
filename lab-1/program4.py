def createA(a1,a2):
    A=[]
    for i in range(a1):
        row=[]
        for j in range(a2):
            a=int(input("Enter element:"))
            row.append(a)
        A.append(row)
    return A

def transpose(A,a1,a2):
    B=[]
    for i in range(a1):
        row1=[]
        for j in range(a2):
            b=A[j][i]
            row1.append(b)
        B.append(row1)
    return B

a1=int(input("Enter number of rows:"))
a2=int(input("Enter number of columns:"))
A=createA(a1,a2)
print("Matrix A:",A)
print("Transpose of Matrix A:",transpose(A,a1,a2))