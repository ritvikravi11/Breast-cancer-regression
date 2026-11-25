import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Drop rows with missing values
    data = data.dropna()
    # Additional cleaning steps can be added here
    return data

def preprocess_data(data):
    # Assuming 'target' is the column name for the target variable
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def save_processed_data(X, y, output_path):
    processed_data = pd.DataFrame(X)
    processed_data['target'] = y
    processed_data.to_csv(output_path, index=False)

def prepare_data(raw_file_path, processed_file_path):
    data = load_data(raw_file_path)
    cleaned_data = clean_data(data)
    X, y = preprocess_data(cleaned_data)
    save_processed_data(X, y, processed_file_path)