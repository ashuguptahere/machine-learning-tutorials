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

# Plotting Graph
plt.scatter(X[y == 0, 0], X[y == 0,1], c = 'r', label = 'Setosa') 
plt.scatter(X[y == 1, 0], X[y == 1,1], c = 'b', label = 'Versicolor')
plt.scatter(X[y == 2, 0], X[y == 2,1], c = 'g', label = 'Virginia')
plt.legend()
plt.show()

plt.scatter(X[y == 0,2], X[y == 0,3], c = 'r', label = 'Setosa')
plt.scatter(X[y == 1,2], X[y == 1,3], c = 'b', label = 'Versicolor')
plt.scatter(X[y == 2,2], X[y == 2,3], c = 'g', label = 'Virginia')
plt.legend()
plt.show()

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

y_pred = log_reg.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Checking Pecision Score, Recall Score, F1 Score
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred, average ='micro')
recall_score(y_test, y_pred, average ='micro')
f1_score(y_test, y_pred, average ='micro')

X = dataset.data[: , 2:]
y = dataset.target

# Plotting Graph
plt.scatter(X[y==0, 0], X[y==0,1], c='r', label = 'Setosa') 
plt.scatter(X[y==1, 0], X[y==1,1], c= 'b', label = 'Versicolor')
plt.scatter(X[y==2, 0], X[y==2,1], c= 'g', label = 'Virginia')
plt.legend()
plt.show()

plt.scatter(X[y==0,2], X[y==0,3], c='r', label = 'Setosa')
plt.scatter(X[y==1,2], X[y==1,3], c= 'b', label = 'Versicolor')
plt.scatter(X[y==2,2], X[y==2,3], c= 'g', label = 'Virginia')
plt.legend()
plt.show()

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split( X, y)

# Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

# Checking Pecision Score, Recall Score, F1 Score
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred, average ='micro')
recall_score(y_test, y_pred, average ='micro')
f1_score(y_test, y_pred, average ='micro')

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth=3)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)

y_pred = dtf.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Exporting Graph
from  sklearn.tree import export_graphviz
export_graphviz(dtf, out_file = "iris.dot")

import graphviz
with open("iris.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)