import random
import statistics

numbers=[random.randint(100,150) for i in range(100)]
mean=statistics.mean(numbers)
median=statistics.median(numbers)
mode=statistics.mode(numbers)
print("Numbers are:",numbers)
print("Mean is:",mean)
print("Median is:",median)
print("Mode is:",mode)