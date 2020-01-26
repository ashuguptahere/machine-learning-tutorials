# Importing Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading Diabetic Dataset
dataset = pd.read_csv('diabetic.csv')

# Dividing Dataset into X (Feature Matrix) and y (Vector of Prediction)
X = dataset.iloc[ : , 0:3].values
y = dataset.iloc[ : , 3].values

# Removing Missing Values with Column's Average using SimpleImputer
from sklearn.impute import SimpleImputer
sim = SimpleImputer()
X[:, 0:2]= sim.fit_transform(X[:, 0:2])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 2] = lab.fit_transform(X[:, 2])
lab.classes_
y[:] = lab.fit_transform(y[:])

# Converting X and y into Pandas Dataframe
X = pd.DataFrame(X)
Y = pd.DataFrame(y)
lab.classes_