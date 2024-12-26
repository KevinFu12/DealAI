import pandas as pd
import numpy as np

def generate_shop_data(num_shops=100, seed=42):
    np.random.seed(seed)
    
    shop_names = [f"Shop_{i+1}" for i in range(num_shops)]
    initial_prices = np.random.uniform(50, 200, num_shops)  # Random initial prices
    selling_prices = initial_prices * (1 + np.random.uniform(0.05, 0.3, num_shops))  # Markup on initial prices
    delivery_times = np.random.randint(1, 15, num_shops)  # Delivery times in days
    
    # Final price determined by negotiation flexibility
    negotiation_flexibility = np.random.uniform(0.8, 1.2, num_shops)
    final_prices = selling_prices * negotiation_flexibility
    
    data = pd.DataFrame({
        'Shop_Name': shop_names,
        'Initial_Price': initial_prices.round(2),
        'Selling_Price': selling_prices.round(2),
        'Delivery_Time': delivery_times,
        'Final_Price': final_prices.round(2)
    })
    
    return data

# Example usage
if __name__ == "__main__":
    shop_data = generate_shop_data()
    shop_data.to_csv("shop_data.csv", index=False)
    print(shop_data.head())
