# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
reshape to vector to matrix
taking 2: to exclude outlier to increse accuracy checked at line 27
Linear Regression helps us to find a linear relation between x and y
f(X)=y type relation . It performs a task of finding a dependent variable y
with help of independent varible X    
"""

# Loading Blood Dataset
dataset = pd.read_excel('blood.xlsx')

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X  = dataset.iloc[ 2: , 1].values
y = dataset.iloc[ 2: , -1].values
X = X.reshape(-1, 1)

# PLotting Graph
plt.scatter(X,y)
plt.show()

# Performing Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X, y)
lin_reg.score(X, y)