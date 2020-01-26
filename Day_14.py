## Image Classification on MNIST Dataset

# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

""" 
we have numbers 0-9 => 10 numbers that we have 
10ary classification 
we will be download using scrapper(work only once)
next we've to use VPN
"""

# Loading dataset
from sklearn.datasets import get_data_home
from sklearn.datasets import fetch_mldata
test_data_home =  get_data_home()
dataset = fetch_mldata('MNIST original', data_home = test_data_home)

dataset = scipy.io.loadmat('mnist-original.mat')

X = dataset['data']
y= dataset['label']
X = X.T
y = y.T
""" 32 is the line number whose image has to shown """

import random
a = random.randint(0,70000)
some_digit  = X[a]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

"""
Shalow Learning
max depth given to tune accuracy
"""

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

# Performing KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)