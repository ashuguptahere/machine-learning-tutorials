# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading Demographics Dataset
dataset = pd.read_csv('DemographicData.csv')
X = dataset.iloc[ : , 2:4 ].values
y = dataset.iloc[ : , -1].values

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
lab.classes_

# Plotting Graph
plt.scatter(X[:,0],X[:,1])
plt.show()

plt.scatter(X[y==0, 0] , X[y == 0 , 1], c = 'r', label = 'High Income')
plt.scatter(X[y==1, 0] , X[y == 1 , 1], c = 'b', label = 'Low income')
plt.scatter(X[y==2, 0] , X[y == 2 , 1], c = 'g', label = 'Lower middle income')
plt.scatter(X[y==3, 0] , X[y == 3 , 1], c = 'y', label = 'Upper middle income')
plt.legend()
plt.xlabel('Birth Rate')
plt.ylabel('Internet Users')
plt.title('Analysis on the WORLD BANK dataset')

plt.show()