import numpy as np

# 4. Reshape arrays into different dimensions.
arr = np.array([2, 4, 6, 8, 10, 12, 14, 16])
print(arr.reshape(2, 4))
print(arr.reshape(4, 2))
print(arr.reshape(8, 1))

# 5. Slice and index arrays.
# indexing
print(arr[4])
print(arr[-1])

# slicing
print(arr[1:5])
print(arr[::2]) 
print(arr[2:])
