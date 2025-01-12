"""
the basic idea of linear regression is to find the best fit line for the data points.
the more data poins a line can hit the better the line is.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
### try to calculate coffecient and interpretor with example and data frame


#df = pd.DataFrame({'x':[0,1,2,3,5,],'y':[ 1, 3, 5 , 7 , 11]})



# 0.99, 3.02, 9.0 , 18.98 , 33.01
df = pd.DataFrame({'x':[0,1,2,3,5,],'y':[ 0.99, 3.02, 9.0 , 18.98 , 33.01]})
print(df.head());

x = df[['x']];
y = df['y'];
model = LinearRegression();
model.fit(x,y);

w1 = model.coef_[0];
w0 = model.intercept_;

y_predict = model.predict(x);
print(y_predict);


## calcultate loss (mostly now people use to get effectiveness of model)
print(round(np.sqrt(mean_squared_error(y, y_predict)), 0))


## calculate score 

print(model.score(x,y));

## plot the model
plt.scatter(x, y, color='blue')  # plot the original data points
plt.plot(x, y_predict, color='red')  # plot the regression line
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Model')
plt.show()