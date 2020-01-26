# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Student Performance Dataset
dataset = pd.read_csv('StudentsPerformance.csv')
dataset.info()
dataset.describe()
dataset.isnull().any
dataset.isnull().sum()

plt.hist(dataset['math score'], bins= 100)
plt.show()

dataset.columns = ['gender','race','ped','lunch','test','math','reading','writing']

sns.barplot(x = dataset['gender'].value_counts().index , y = dataset['gender'].value_counts())

sns.barplot(dataset['race'] , dataset['math'],hue=dataset['gender'])
sns.barplot(dataset['ped'] , dataset['math'],hue=dataset['gender'])
sns.barplot(dataset['test'] , dataset['math'],hue=dataset['gender'])
sns.barplot(dataset['lunch'] , dataset['math'],hue=dataset['gender'])

sns.boxplot(dataset['math'])
sns.boxplot(dataset['writing'])
sns.boxplot(dataset['reading'])

sns.boxplot(dataset['gender'],dataset['math'])
sns.boxplot(dataset['gender'],dataset['writing'])
sns.boxplot(dataset['gender'],dataset['reading'])

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

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