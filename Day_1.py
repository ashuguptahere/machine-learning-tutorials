# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5])
b = np.array([[1,2,3] ,
             [4,5,6] , 
             [7,8,9] ])￼

# Loading Dataset
dataset = pd.read_csv('day1.csv')
plt.scatter([1,2,3],[3,2,1])

# Plotting Graph
plt.scatter(dataset['raw'], np.zeros(10), marker = 'X')
plt.scatter(dataset['fertilised'], np.ones(10), marker = 'o')

plt.hist(dataset['fertilised'])
plt.hist(dataset['raw'])

plt.scatter( dataset['raw'], dataset['fertilised'] , marker = 'o' , color = 'r')
plt.show()

y = pd.Series([1,2,3,4,5] , [45,85,778,889,991])
x = pd.DataFrame({1 : [1,2,3,4,5] , 
	              2 : [6,7,8,9,10] ,
	              3 : [11,12,13,14,15] ,
	              4 : [16,17,18,19,20] ,
	              5 : [21,22,23,24,25]})

x.iloc[2:5, 3: ].values
x.iloc[3:, 3: ].values

s = np.zeros((3,4),dtype = 'int')
d = np.full((3, 3), 5 , dtype = 'complex')