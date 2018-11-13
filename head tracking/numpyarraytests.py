import numpy as np


#defining arrays
#https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.array.html

x = np.array([[ 68,  87,  24 , 24], [157 , 61 , 61  ,61], [ 49  ,71 , 56,  56], [141  ,70  ,51 , 51]])
#arrays within arrays
print(x[2][2]) #arrays are zero indexed so this is the 3rd array, 3rd ellement

#ndarray atributes
#https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.html

print(x.T) #transpose

print(x.size) #number elements in the array

