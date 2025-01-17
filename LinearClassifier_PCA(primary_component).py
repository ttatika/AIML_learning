import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions # type: ignore
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Select features and target (binary classification: setosa vs. non-setosa)
df = df[df['species'] != 'virginica']  # Remove one class to make it binary
X = df.drop(columns=['species']).values
y = (df['species'] == 'setosa').astype(int).values  # Binary target: 1 if setosa, else 0
df.dropna()
# Ensure X and y have the same number of samples
assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SGDClassifier with hinge loss for SVM, a maximum of 1000 iterations, and a tolerance of 0.001 for early stopping
model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
print(model.predict(X_test))
print(y_test)

# Apply PCA to reduce the dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the decision boundary using the PCA-transformed features
plot_decision_regions(X_pca, y, clf=model, legend=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SGDClassifier Decision Boundary with PCA')
plt.show()