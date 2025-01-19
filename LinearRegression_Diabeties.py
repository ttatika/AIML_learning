#Linear regresion on Diabeties Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  .metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging
#label encoder
from sklearn.preprocessing import LabelEncoder
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# get diabetes dataset
def load_data():
    """Load the dataset."""
    # get from online
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    df = pd.read_csv(url)
    logging.info("Data loaded successfully")
    return df

def train_model(x, y):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(x, y)
    logging.info("Model trained successfully")
    return model

def evaluate_model(model, x, y):
    """Evaluate the model and print metrics."""
    y_predict = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    logging.info(f"Root Mean Squared Error: {rmse:.2f}")
    return y_predict, rmse

def plot_model(x, y, y_predict):
    """Plot the data points and the regression line."""
    plt.scatter(x, y, color='blue')  # plot the original data points
    plt.plot(x, y_predict, color='red')  # plot the regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Model')
    plt.show()
    logging.info("Model plot generated successfully")

def main():
    """Main function to run the linear regression example."""
    df = load_data()
    #label encode the target
    label_encoder = LabelEncoder()
    df['Outcome'] = label_encoder.fit_transform(df['Outcome'])
    x = df[['Glucose']]
    y = df['Outcome']
    
    model = train_model(x, y)
    y_predict, rmse = evaluate_model(model, x, y)
    
    logging.info(f"Model Coefficient: {model.coef_[0]}")
    logging.info(f"Model Intercept: {model.intercept_}")
    
    plot_model(x, y, y_predict)

if __name__ == "__main__":
    main()
# Compare this snippet from LogisticRegression.py:
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

