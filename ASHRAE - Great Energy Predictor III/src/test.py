import numpy as np
a = [[1,2,3,4], [2,3,4,5], [5,6,7,8]]
print(a)

b = np.array(a)
print(b)
print(b.shape)

print(b.mean(axis=0))
print(b.mean(axis=1))
print(8/3, 11/3)