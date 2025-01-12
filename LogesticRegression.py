import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Example data
df = pd.DataFrame({'x':[0, 1, 2, 3, 5], 'y':[0, 0, 1, 1, 1]})
print(df.head())

x = df[['x']]
y = df['y']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)
print(y_predict)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# Plot the data points and decision boundary
plt.scatter(x, y, color='blue')  # plot the original data points
x_values = np.linspace(min(x['x']), max(x['x']), 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_values, color='red')  # plot the decision boundary
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Logistic Regression Model')
plt.show() 