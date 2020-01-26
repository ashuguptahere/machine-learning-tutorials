"""
y = Vector of prediction ---> Dependent variable 
X = Feature matrix ---> Independent Variable
Normal Equation : line 29 ---> @ means dot product 
normal equation helps to find the line of prediction (Lop)
Lop is based on OLS techinque (ordinary least square technique)
lop = min of square of euclidian distance
"""

# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X (Feature Matrix) and y (Vector of Prediction)
X = np.random.randn(100)
y = 8 + 3*X + np.random.randn(100)

plt.plot(X , y)
plt.show()

plt.scatter(X , y)
plt.show()

# Adding a Column of ones in X Matrix
X = np.c_[X , np.ones(100)]

theta = np.linalg.inv( X.T @ X) @ (X.T @ y)