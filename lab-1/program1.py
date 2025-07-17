def findvowels(string,vowels):
    count1=0
    for i in string:
        if(i.lower() in vowels):
            count1+=1
    return count1

def findconsonants(string,vowels):
    count2=0
    for i in string:
        if(i.lower() not in vowels):
            count2+=1
    return count2

vowels=['a','e','i','o','u']
string=input("Enter string:")
print("Number of vowels:",findvowels(string,vowels))
print("Number of consonants:",findconsonants(string,vowels))