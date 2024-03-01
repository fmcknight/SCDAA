import numpy as np
from random import random

sum = []
count = 0
n=10000

while len(sum)<n:
    
    x = random()
    y = random()
    if (x**2+y**2)<=1:
        sum.append(np.sqrt(x**2+y**2))
    else:
        count+=1

print(np.mean(sum))
print(count/n)
print(1-np.pi/4)