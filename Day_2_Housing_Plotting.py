# Importing Essential Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Loading Housing Dataset
dataset = pd.read_csv('housing.csv')
scatter_matrix(dataset)

# Plotting Graph
plt.scatter(dataset['total_rooms'], dataset['total_bedrooms'])
plt.show()

plt.scatter()
x = np.arange(-10, 10, 0.01)
y = 0.7 * x + 5
plt.plot(x, y)
plt.show()

y1 = 0.7 * x ** 2 + x + 8
plt.plot(x, y1)
plt.show()

sig_y = 1 / (1 + np.power(np.e, -x))
plt.plot(x, sig_y)
plt.show()

a = np.random.randn(10)
b = np.random.randn(5, 5)

pd.scatter_matrix(dataset.loc[:, :]) 
pd.show_versions(as_json = False)

corr_mat = dataset.corr()
sns.heatmap(corr_mat, annot = True)

np.arange(23, 55, 2)
np.linspace(0, 100, 6)