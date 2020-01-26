# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Loading IRIS dataset
from sklearn.datasets import load_iris
dataset= load_iris()
X = dataset.data
y = dataset.target

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Performing Support Vector Classification
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_train, y_train)

# Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)

# Performing Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
n_b.score(X, y)

# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)