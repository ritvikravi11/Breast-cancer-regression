from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Assuming the target variable is named 'target' and features are all other columns
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y

def train_model(X_train, y_train=None):
    if y_train is None:
        target_column = 'likelihood_of_cancer' if 'likelihood_of_cancer' in X_train.columns else 'cancer_likelihood'
        if target_column not in X_train.columns:
            raise KeyError("Training data must contain a cancer likelihood target column")
        y_train = X_train[target_column]
        X_train = X_train.drop(columns=[target_column])
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def main():
    # Load and preprocess data
    data = load_data('../data/raw/custom_dataset.csv')
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    main()