"""
this file is mostly to show the scalling technique on data to standerdize the data.
explanation :
The StandardScaler from sklearn.preprocessing is used to standardize features by removing the mean and scaling to unit variance. This means that it transforms the data such that the distribution has a mean of 0 and a standard deviation of 1.

Here's a step-by-step explanation of what the StandardScaler does:

Fit the Scaler: Calculate the mean and standard deviation for each feature in the training set.
Transform the Data: Subtract the mean and divide by the standard deviation for each feature.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
scaler = StandardScaler()
scaler.fit(data)
print(data)
data = scaler.fit_transform(data)
print(data)