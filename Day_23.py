# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Making Fake Data
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.6)

plt.scatter(x[:,0], x[:,1])
plt.show()

# Performing KMeans Clustering
from sklearn.cluster import KMeans

wcv =[]

for i in range (1,15): 
    km = KMeans(n_clusters = i)
    km.fit(x)    
    wcv.append(km.inertia_)
    
plt.plot(range(1,15), wcv)
plt.show()

km = KMeans(n_clusters = 5)
y_pred = km.fit_predict(x)

# Plotting the points onto the graph using MatPlotLib
plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1])
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1])
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1])
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1])
plt.scatter(x[y_pred == 4,0], x[y_pred == 4,1])
plt.show()

for i in range(0,5):
    plt.scatter(x[y_pred==i,0], x[y_pred==i,1])
plt.show()