import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import os

# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def find_best_shop_knn(data, sample_input, k=3):
    features = ['Initial_Price', 'Selling_Price', 'Delivery_Time', 'Final_Price']
    
    # fill empty with mean values
    sample_input = sample_input[0]
    for i, feature in enumerate(features):
        if sample_input[i] == 0 or sample_input[i] == "":
            sample_input[i] = data[feature].mean()
            
    sample_input = [sample_input]
    
    # Prepare data for k-NN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # Fit k-NN model
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(scaled_data)

    # Scale sample input
    sample_scaled = scaler.transform(sample_input)

    # Find nearest neighbor
    distances, indices = knn.kneighbors(sample_scaled)
    nearest_shops = data.iloc[indices[0]]

    return nearest_shops

# Evaluate model
def evaluate_model(data, predicted_output):
    mse = mean_squared_error(data['Final_Price'], predicted_output)
    return mse

if __name__ == "__main__":
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'shop_data.csv')  
    shop_data = load_data(file_path)
    
    # Dynamic Input
    features = shop_data.select_dtypes(include=['int64', 'float64']).columns
    
    sample_input = []
    for feature in features:
        val = input(f"Enter {feature} (Empty if none): ")
        if val == "":
            sample_input.append(shop_data[feature].mean())
        else:
            sample_input.append(float(val))
            
    sample_input = [sample_input]
    best_shops = find_best_shop_knn(shop_data, sample_input)

    print(f"Sample input: {sample_input}")
    print("3 Best Offer:")
    print(best_shops)