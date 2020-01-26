# Importing Essential Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from mpl_toolkits import mplot3d

# Loading Boston Dataset
from sklearn.datasets import load_boston
dataset = load_boston()

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.data
y = dataset.target

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Performing Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)
lin_reg.score(X, y)

y_pred = lin_reg.predict(X_test)

# Plotting Graph
plt.scatter(X[:,2], y)
plt.show()

sns.pairplot()
plt.plot(X_test, y_pred, c = 'g')
plt.show()

plt.scatter(y_test, y_pred)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices$")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

fig = plt.figure()
ax = plt.axes(projection = '3d')
a = np.cos(2*np.pi*np.random.rand(0, 2))