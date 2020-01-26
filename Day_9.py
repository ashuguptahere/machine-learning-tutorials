# Importing Essential Libraries
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

# Loading Blood Dataset
dataset = pd.read_excel('blood.xlsx')

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.iloc[ : , 1].values
y = dataset.iloc[ : , -1].values
X = X.reshape(-1,1)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(X,y)

# Performing Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X_train , y_train)

y_pred =lin_reg.predict(X_test)

lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)
lin_reg.score(X,y)

# Plotting Graph
plt.scatter(X_test, y_test)
plt.scatter(X_train, y_train)

plt.plot(X, lin_reg.predict(X), c= 'y')
plt.show()

""""
line 27 predicts the LOP and its slope is lin_reg.coef_ and intercept is 
 lin_reg.intercept_ lin_reg.predict([[X]]) act as fuction and X is the 
 input given which calaculates according to the LOP and gives output
 test and train 75% and 25%  randomly everytime takes different values
"""

lin_reg.coef_
lin_reg.intercept_

lin_reg.predict([[26]])

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
export_graphviz(dtf, out_file = "blood.dot")

import graphviz
with open("blood.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)