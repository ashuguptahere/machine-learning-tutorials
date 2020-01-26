# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading IRIS Dataset
from sklearn.datasets import load_iris
dataset = load_iris()

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.data
y = dataset.target

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test,y_test)
dtf.score(X,y)

'''install graphviz'''