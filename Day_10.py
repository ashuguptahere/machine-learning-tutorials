# Importing Essential Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Making Fake X nd y
m = 100
X = 6*np.random.randn(m,1)-3 
y = 0.5*X**3 +3*X*X+6*X +4 + np.random.randn(m,1)

plt.scatter(X,y)
plt.axis([-5,5,-10,10])
plt.show()

# Performing Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Performing Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X_poly, y)

X_new = np.linspace(-5,5,100).reshape(-1,1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)

# Plotting Graph
plt.scatter(X,y)
plt.plot(X_new, y_new, c="g")
plt.axis([-5,5,-10,10])
plt.show()

lin_reg.predict([[3,9,27]])

# Coefficient Beta-0
lin_reg.coef_

# Intercept Beta-1
lin_reg.intercept_