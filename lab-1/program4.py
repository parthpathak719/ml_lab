a1=int(input("Enter number of rows:"))
a2=int(input("Enter number of columns:"))
A=[]
for i in range(a1):
    row=[]
    for j in range(a2):
        a=int(input("Enter element:"))
        row.append(a)
    A.append(row)
print("Matrix A:",A)

B=[]
for i in range(a1):
    row1=[]
    for j in range(a2):
        b=A[j][i]
        row1.append(b)
    B.append(row1)
print("Transpose of Matrix A:",B)