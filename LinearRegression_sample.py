import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

### try to calculate coffecient and interpretor with example and data frame

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