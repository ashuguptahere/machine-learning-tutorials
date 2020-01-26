# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Making Fake Data
from sklearn.datasets import make_blobs
x,y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.6)

plt.scatter(x[:,0], x[:,1])
plt.show()

# Making a Dendrogram (Tree Like Structure) [Mathematics Topic] using SciPy Library
import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x, method = 'complete'))

# Performing HCA
from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters = 8)
y_pred = hca.fit_predict(x)

# Plotting 
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0,1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1,1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2,1])
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3,1])
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4,1])
plt.scatter(x[y_pred == 5, 0], x[y_pred == 5,1])
plt.scatter(x[y_pred == 6, 0], x[y_pred == 6,1])
plt.scatter(x[y_pred == 7, 0], x[y_pred == 7,1])
plt.show()

from apyori import apriori

'''
wcv =[]

for i in range (1,15): 
    hca = AgglomerativeClustering(n_clusters=i)
    hca.fit(x)    
    wcv.append(hca.)
    
plt.plot(range(1,15), wcv)
plt.show()

'''