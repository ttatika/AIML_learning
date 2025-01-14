import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent
 

 ## download the pengiun dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')
print(df.head())
df = df[df['species'] != 'Gentoo']
## drop the null values
df = df.dropna()
print(df.head())
print(df.shape)
df.drop('island', axis=1, inplace=True);
df.drop('sex', axis=1, inplace=True);

# use label encoding to convert categorical data to numerical data
# this is necessary because the model can only work with numerical data (output is alphabatical)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])
print(df.head())


# label encode Biscoe, Dream, Torgersen to 0, 1, 2

print(df['species'].unique())
print(label_encoder.inverse_transform(df['species'].unique())) # reverse of label encoding

# Initialize the SGDClassifier with hinge loss for SVM, a maximum of 1000 iterations, and a tolerance of 0.001 for early stopping
model = SGDClassifier(loss='hinge');
#
# split in test and train
X = df.drop(columns=['species']).values
y = df['species'].values
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train);
model.score(X_test, y_test);
print(model.score(X_test, y_test))
print(model.predict(X_test))
print(y_test)
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# plot the decision boundary
from mlxtend.plotting import plot_decision_regions # type: ignore
plot_decision_regions(X, y, clf=model, legend=2)

# plot using matplotlib
import matplotlib.pyplot as plt
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.title('Penguin Species Classification')
plt.show()