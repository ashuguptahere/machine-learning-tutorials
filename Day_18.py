'''
Natural Language Preprocessing

pip install nltk
nltk.download('stopwords')
'''
# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Importing Dataset
dataset = pd.read_csv('train.csv')
dataset['tweet'][0]
processed_tweet = []
#removing Symbols keeping only string and #
""" 
    re is redistribution to operate strings with operations like trimming etc
    sub (1,2,3) 
    1- to be replaced 
    2- repalacement 
    3- where to perform
    For loop
    line 1 @ followed by word every where 
    line 2 not a-zA-Z# not removed bur everything else replace by " "
    line 3 everything to lower case
    line 4 spilt whole string into a list containing those elements
    line 5 remove all unwanted words preposition conjunction etc
    line 5 again join into a string when ' ' appears performs joining opereation 
    followed 
    line 6 save into a  list conatining every tweet in list
"""
for i in range (31962):
    temp = re.sub('@[\w]*', ' ', dataset['tweet'][i]) 
    temp = re.sub('^a-zA-Z#', ' ', temp)
    temp = temp.lower()
    temp = temp.split()
    temp = [ps.stem(token) for token in temp if not token in set(stopwords.words('english'))]
    temp = ' '.join(temp)
    processed_tweet.append(temp)

# Performing CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(processed_tweet)
X = X.toarray()
y = dataset['label'].values
print(cv.get_feature_names)

# Train-Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Performing Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
n_b.score(X_train, y_train)

# Performing KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
    
# Performing Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth=3)
dtf.fit(X_train, y_train)
dtf.score(X_train, y_train)