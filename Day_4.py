# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading Diabetic Dataset
dataset = pd.read_csv('diabetic.csv')
X = dataset.iloc[ : , 0:3].values
y = dataset.iloc[ : , 3  ].values

# Removing Missing Values with Column's Average using SimpleImputer
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
sim = SimpleImputer()
X[:, 0:2] = sim.fit_transform(X[: , 0:2])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 2 ] = lab.fit_transform(X[:, 2])
lab.classes_
y[:] = lab.fit_transform(y[:])

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X).toarray()

# Scaling the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = lib.fit_transform(y)
lib.classes_

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split( X, y)

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth=3)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)

y_pred = dtf.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Exporting and Reading the Graph
from  sklearn.tree import export_graphviz
export_graphviz(dtf, out_file = "std.dot")

import graphviz
with open("std.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)