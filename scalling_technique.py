"""
This file is mostly to show the scaling technique on data to standardize the data.
Explanation:
The StandardScaler from sklearn.preprocessing is used to standardize features by removing the mean and scaling to unit variance. This means that it transforms the data such that the distribution has a mean of 0 and a standard deviation of 1.

Here's a step-by-step explanation of what the StandardScaler does:

Fit the Scaler: Calculate the mean and standard deviation for each feature in the training set.
Transform the Data: Subtract the mean and divide by the standard deviation for each feature.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data (calculate mean and std)
scaler.fit(data)

# Transform the data (standardize it)
scaled_data = scaler.transform(data)

# Plot the original data
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data')

# Plot the scaled data
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], color='red', label='Scaled Data')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original vs Scaled Data')
plt.legend()

# Show the plot
plt.show()