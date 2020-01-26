# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.data
y = dataset.target

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y) 

# Performing KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X,y)

y_pred = knn.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)