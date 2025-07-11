vowels=['a','e','i','o','u']
count1=0
count2=0
string=input("Enter string:")
for i in string:
    if(i.lower() in vowels):
        count1+=1
    else:
        count2+=1
print("Number of vowels:",count1)
print("Number of consonants:",count2)