import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Fill empty fields with mean values
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

    return nearest_shops, scaled_data, scaler, indices

# Evaluate model
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return mse

def plot_knn_relationship(data, sample_input, nearest_shops, indices, scaler):
    plt.figure(figsize=(10, 6))

    # Scatter plot of Initial_Price vs Final_Price for the entire dataset
    sns.scatterplot(data=data, x='Initial_Price', y='Final_Price', color='blue', label='All Shops')

    # Highlight the nearest shops (3 best offers)
    nearest_shops_scaled = data.iloc[indices[0]]
    sns.scatterplot(data=nearest_shops_scaled, x='Initial_Price', y='Final_Price', 
                    color='red', s=100, label='Best Shops')

    # Plot the sample input (the one provided by the user)
    sample_input_scaled = scaler.inverse_transform([scaler.transform([sample_input[0]])[0]])
    plt.scatter(sample_input_scaled[0][0], sample_input_scaled[0][-1], color='green', 
                label='Sample Input', s=200, edgecolor='black', marker='D')

    # Add labels and title
    plt.title('K-NN: Harga Awal vs Harga Akhir', fontsize=16)
    plt.xlabel('Harga Awal', fontsize=14)
    plt.ylabel('Harga Akhir', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

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
    
    # Find Best Shops using k-NN
    best_shops, scaled_data, scaler, indices = find_best_shop_knn(shop_data, sample_input)
    
    # Print out the best offers
    print(f"Sample input: {sample_input}")
    print("3 Best Offer:")
    print(best_shops)

    # Evaluate the Model:
    # Get the 'Final_Price' of the best 3 shops for evaluation
    predicted_prices = best_shops['Final_Price'].mean()  # Take the mean of the 'Final_Price' from the top 3 shops
    actual_price = sample_input[0][-1]  # Assuming the last input value corresponds to 'Final_Price' (target)
    
    # Evaluate the model
    mse = evaluate_model([actual_price], [predicted_prices])

    # Output the results
    print(f"Predicted Average Final Price from Best Shops: {predicted_prices}")
    print(f"Actual Final Price: {actual_price}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Plot the K-NN relationship for Initial_Price vs Final_Price
    plot_knn_relationship(shop_data, sample_input, best_shops, indices, scaler)
