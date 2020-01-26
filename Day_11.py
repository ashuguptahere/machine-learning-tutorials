# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Linear Regression
X = np.arange(-10,10,0.01)

line = - 4 - 3*X

y = 1/(1 + np.power( np.e , line))
plt.plot(X, y)
plt.show()

sig_y = 1/(1 + np.power( np.e , -X))
plt.plot(X, sig_y)
plt.show()