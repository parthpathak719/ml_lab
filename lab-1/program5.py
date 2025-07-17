import random
import statistics

def findmean(numbers):
    mean=statistics.mean(numbers)
    return mean

def findmedian(numbers):
    median=statistics.median(numbers)
    return median

def findmode(numbers):
    mode=statistics.mode(numbers)
    return mode

numbers=[random.randint(100,150) for i in range(100)]
print("Numbers are:",numbers)
print("Mean is:",findmean(numbers))
print("Median is:",findmedian(numbers))
print("Mode is:",findmode(numbers))