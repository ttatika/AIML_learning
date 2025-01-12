import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Select features and target
X = df.drop(columns=['species']).values
y = df['species'].values

# Ensure X and y have the same number of samples
assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# Plot the original data points (using the first two features for visualization)
plt.scatter(X[:, 0], X[:, 1], c=pd.factorize(y)[0], cmap='viridis', label='Original Data')

# Plot the test points and their neighbors
for i, test_point in enumerate(X_test):
    neighbors = model.kneighbors([test_point], return_distance=False)
    for neighbor in neighbors[0]:
        plt.plot([test_point[0], X_train[neighbor, 0]], [test_point[1], X_train[neighbor, 1]], 'k--')
    circle = plt.Circle((test_point[0], test_point[1]), radius=0.5, color='r', fill=False)
    plt.gca().add_patch(circle)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classification with Neighbors')
plt.legend()
plt.show()