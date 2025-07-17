def createA(s1):
    A=[]
    for i in range(s1):
        a=int(input("Enter integer element for A:"))
        A.append(a)
    return A

def createB(s2):
    B=[]
    for i in range(s2):
        b=int(input("Enter integer element for B:"))
        B.append(b)
    return B

def countcommon(A,B):
    count=0
    for i in A:
        if i in B:
            count+=1
    return count

s1=int(input("Enter size of list A:"))
s2=int(input("Enter size of list B:"))
A=createA(s1)
B=createB(s2)
print("Number of common elements:",countcommon(A,B))