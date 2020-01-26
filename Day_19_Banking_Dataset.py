# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Dataset
dataset = pd.read_csv('bank.csv')

sns.barplot(x=dataset['job'].value_counts().index , y=dataset['job'].value_counts())

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:,-1].values

dataset.isnull().sum()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:, 1] = lab.fit_transform(X[:, 1])
X[:, 2] = lab.fit_transform(X[:, 2])
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 4] = lab.fit_transform(X[:, 4])
X[:, 5] = lab.fit_transform(X[:, 5])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 9] = lab.fit_transform(X[:, 9])
X[:, 14] = lab.fit_transform(X[:, 14])

# Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Fitting the data and checking score
log_reg.fit(X_train, y_train)
log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)
log_reg.score(X,y)

# Performing KNN Classification
from  sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 
knn.fit(X_train, y_train)
knn.score(X_train, y_train)

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth=3)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)

y_pred = dtf.predict(X_test)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Exporting the graphviz file
from  sklearn.tree import export_graphviz
export_graphviz(dtf, out_file = "banking.dot")

# Reading the Graph made by Decision Tree Classifier
import graphviz
with open("bank.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)