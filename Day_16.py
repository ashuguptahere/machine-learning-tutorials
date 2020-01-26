# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

# Importing Load Digits Dataset
from sklearn.datasets import load_digits

# Importing MNIST Dataset
dataset = scipy.io.loadmat('mnist-original.mat')

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset['data']
y= dataset['label']
X = X.T
y = y.T

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth=3)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test,y_test)
dtf.score(X,y)

y_pred = dtf.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

dtf.predict(X[[5600, 7000],0:784])

# GraphVIZ
from sklearn.tree import export_graphviz
export_graphviz(dtf, out_file = "tree.dot")

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
dot_data = StringIO()
export_graphviz(dtf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())