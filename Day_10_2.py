# Importing Essential Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading Housing Dataset
from sklearn.datasets import load_boston
dataset = pd.read_csv("housing.csv")
dataset.isnull().sum()

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.iloc[ : , [0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[ : , -2 ].values

# Removing Missing Values with Column's Average using SimpleImputer
from sklearn.impute import SimpleImputer
sim = SimpleImputer()
X[:, [0,1,2,3,4,5,6,7]] = sim.fit_transform(X[:, [0,1,2,3,4,5,6,7]])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab =  LabelEncoder()
X[: , -1] = lab.fit_transform(X[: , -1])
lab.classes_

dataset.describe()
dataset.info()

# Pandas Scatter Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split( X, y, test_size=0.1)

# Performing Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X, y)
lin_reg.score(X, y)
lin_reg.fit(X_train, y_train)
lin_reg.score(X_train, y_train)

plt.scatter(X_test[:, 1], y_test)
plt.show()

lin_reg.fit(X_test, y_test)
lin_reg.score(X_test, y_test)
