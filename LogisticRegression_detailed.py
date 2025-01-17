'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Load the Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

df.dropna()
# Select features and target
X = df.drop(columns=['species']).values
y = df['species'].values




#label encode the target
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#print the x header
print(df.head())

# Ensure X and y have the same number of samples
assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')

#plot the logestic regression classification based on the model created on overall data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classification')
plt.legend()
plt.show()

## plot decision boundary region with 4 features
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=model, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()





#plot the results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classification')
plt.legend()
plt.show()'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Select features and target
X = df.drop(columns=['species']).values
y = df['species'].values
# label encoder for y
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Ensure X and y have the same number of samples
assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Accuracy: {accuracy}')
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Apply PCA to reduce the dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the decision boundary using the PCA-transformed features
plot_decision_regions(X_pca, y, clf=model, legend=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression Decision Boundary with PCA')
plt.show()