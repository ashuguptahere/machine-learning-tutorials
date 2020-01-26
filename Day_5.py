# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading Salary Dataset
dataset = pd.read_csv('sal.csv', names = ['age','work-class','fnlwgt','education',
                                          'education-num','marital-status','occupation',
                                          'relationship','race','gender','capital-gain',
                                          'capital-loss','hours-perweek','native-country','salary'],
                                            na_values= ' ?')

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.iloc[ : , 0:14].values
y = dataset.iloc[ : , -1].values

# Removing Missing Values with Column's Average using SimpleImputer
from sklearn.impute import SimpleImputer
sim = SimpleImputer()
X[: , [0,2,4,10,11,12]]= sim.fit_transform(X[: , [0,2,4,10,11,12]])

""" 
since string column can not be operated as we did in line number 16 so we assign them to a 
new vcariable
"""
test = pd.DataFrame( X[ : , [1,3,5,6,7,8,9,13] ] )
test[0].value_counts()
test[1].value_counts()
test[2].value_counts()
test[3].value_counts()
test[4].value_counts()
test[5].value_counts()
test[6].value_counts()
test[7].value_counts()

test[0] = test[0].fillna(' Private')
test[1] = test[1].fillna(' HS-grad')
test[2] = test[2].fillna(' Married-civ-spouse')
test[3] = test[3].fillna(' Prof-specialty')
test[4] = test[4].fillna(' Husband')
test[5] = test[5].fillna(' White')
test[6] = test[6].fillna(' Male')
test[7] = test[7].fillna(' United-States')

""" After that missing values are filled """

X[:, [1,3,5,6,7,8,9,13] ] = test

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab  = LabelEncoder()
X[ : , 1] = lab.fit_transform(X[ :, 1])
X[ : , 3] = lab.fit_transform(X[ :, 3])
X[ : , 5] = lab.fit_transform(X[ :, 5])
X[ : , 6] = lab.fit_transform(X[ :, 6])
X[ : , 7] = lab.fit_transform(X[ :, 7])
X[ : , 8] = lab.fit_transform(X[ :, 8])
X[ : , 9] = lab.fit_transform(X[ :, 9])
X[ : , 13] = lab.fit_transform(X[ :, 13])
lab.classes_

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder( categorical_features=[ 1, 3, 5, 6, 7, 8, 9, 13])
X = one.fit_transform(X)
X = X.toarray()
y = lab.fit_transform(y)

# Scaling the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Performing KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)

y_pred = knn.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)