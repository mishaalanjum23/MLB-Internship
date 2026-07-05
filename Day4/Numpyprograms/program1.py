import numpy as np

#  1. Create 1D and 2D arrays. 
array = np.array([10, 20, 30])
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(array)
print(matrix)

# 2. Perform arithmetic operations on arrays.
a = np.array([2, 4, 6])
b = np.array([1, 3, 5])
print(a + b)
print(a - b)
print(a * b)
print(a / b) 

# 3. Find the maximum, minimum, mean, and sum of an array.
print("Maximum value of array is", np.max(array))
print("Minimum value of array is", np.min(array))
print("Mean of array is", np.mean(array))
print("Sum of array is", np.sum(array))