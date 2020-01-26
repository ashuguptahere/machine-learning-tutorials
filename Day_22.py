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

# Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# Performing Support Vector Classification
from sklearn.svm import SVC
svm = SVC()

# Performing Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

# Performing KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
    
# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

# Performing Ensemble Learning - Soft Voting Classification
from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('LR', log_reg),
                       ('SVM', svm),
                       ('NB',n_b),
                       ('KNN', knn),
                       ('DT',dtf)], voting = 'Soft')
vot.fit(X_train, y_train)
vot.score(X_train, y_train)

# Performing Ensemble Learning - hard Voting Classification
from sklearn.ensemble import VotingClassifier
vot2 = VotingClassifier([('LR', log_reg),
                       ('SVM', svm),
                       ('NB',n_b),
                       ('KNN', knn),
                       ('DT',dtf)], voting = 'hard')
vot2.fit(X_train, y_train)
vot2.score(X_train, y_train)

# Performing Ensemble Learning
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(log_reg, n_estimators = 5)
bag.fit(X_train, y_train)
bag.score(X_train, y_train)

# Performing Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(n_estimators = 5)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)