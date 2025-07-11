s1=int(input("Enter size of list A:"))
s2=int(input("Enter size of list B:"))
A=[]
B=[]
for i in range(s1):
    a=int(input("Enter integer element for A:"))
    A.append(a)
for i in range(s2):
    b=int(input("Enter integer element for B:"))
    B.append(b)
print("List A:",A)
print("List B:",B)

count=0
for i in A:
    if i in B:
        count+=1
print("Number of common elements:",count)