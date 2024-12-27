#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\offerBot\Scripts\Activate.ps1

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import importlib.util
import os
import math
import re
from sklearn.utils.validation import check_array
import warnings
import nltk
from nltk.tokenize import word_tokenize

def detect_bargaining(input_text):
    tokens = word_tokenize(input_text.lower())
    bargaining_keywords = ["", "maybe", "buy", "sell", "offer", "price", "about", "dunno"]
    bargaining_verbs = ["buy", "sell"]

    # only digits
    if len(tokens) == 1 and tokens[0].isdigit():
        return True

    for token in tokens:
        if token in bargaining_keywords:
            return True
        if token in bargaining_verbs and tokens[0] == "i":
            return True

    return False

warnings.filterwarnings("ignore")

# Menghilangkan peringatan
def ignore_warnings(*args, **kwargs):
    pass

check_array = ignore_warnings

# Impor BargainModel
spec = importlib.util.spec_from_file_location("BargainModel", "BargainModel.py")
BargainModel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BargainModel)

def select_random_shop(shop_data):
    random_shop = shop_data.sample(n=1)
    return random_shop

# Find 3 best shops
def find_best_shops(shop_data, sample_input):
    return BargainModel.find_best_shop_knn(shop_data, sample_input, k=3)

def determine_response(user_input, best_shops):
    # Extract the user price from the input (assumed to be the last number in the sentence)
    user_price = int(user_input.split()[-1])

    # Find the closest price to the user price
    best_shops['Price_Difference'] = abs(best_shops['Final_Price'] - user_price)
    closest_shop = best_shops.loc[best_shops['Price_Difference'].idxmin()]
    
    # Retrieve the shop details for the closest shop
    lowest = math.ceil(closest_shop['Final_Price'])
    highest = math.ceil(best_shops['Final_Price'].max())
    delivery_time = closest_shop['Delivery_Time']
    shop_name = closest_shop['Shop_Name']

    # Determine the response based on the user's price
    if user_price <= 0.8 * lowest:
        # Reject
        prompt = f"The offer is rejected, the highest I can do is {lowest}. Sorry!"
    elif user_price < lowest:
        # Counteroffer
        prompt = f"Counteroffer: {math.ceil(1.01 * lowest)}, I think this is more competitive. Delivery in {delivery_time} days from {shop_name}."
    elif user_price >= lowest and user_price <= highest:
        # Accept
        prompt = f"Accept! Your book will arrive in {delivery_time} days from {shop_name} for {user_price}! Do you accept?"
    elif user_price > highest:
        prompt = f"OF COURSE, ACCEPT. Thank You! Delivery in {delivery_time} days from {shop_name} for {user_price}. Do you accept?"

    return prompt

def extract_numbers(user_input):
    return [int(num) for num in re.findall(r'\d+', user_input)]

# Main program
if __name__ == "__main__":
    # Load shops data
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'shop_data.csv')
    shop_data = BargainModel.load_data(file_path)

    # Initiate
    template = """
    Answer the question below.

    Here is the conversation history: {context}
    Question: {question}
    Answer:
    """
    model  = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    random_shop = select_random_shop(shop_data)
    offer_price = math.ceil(random_shop['Selling_Price'].values[0])
    initial_price = random_shop['Initial_Price'].values[0]

    context = ""
    print("Welcome, I am DealAI, type 'exit' to quit")
    res = chain.invoke({"context": "", "question": f"You are DealAI, an e-commerce bargain bot. First, crack a dad joke or a pun. You are not allowed to talk about anything other than bargaining. Keep that in mind. First you introduce yourself. Explain that you have 1 interesting modern novel to offer with price of around {offer_price} thousand rupiahs, tell the user the title and the story"})
    print(res)

    best_shops = None
    
    affirmative_responses = {"deal", "yeah","ye", "y","yea", "ya", "yes", "i think so", "i do", "uh huh"}
    rejection_responses = {"nope", "no", "n", "dont", "goodbye", "dunno", "nuh uh", "nah"}
    farewell_responses = {"thanks", "im good", "no thanks", "thank you", "thank", "much obliged", "see you"}

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Thank you for using DealAI, have a nice day.")
            break
        
        if any(farewell in user_input.lower() for farewell in farewell_responses):
            print("Bot: Okay, thank you for using DealAI, have a nice day.")
            break

        numbers = extract_numbers(user_input)

        if detect_bargaining(user_input):
            if best_shops is None:  
                sample_input = [[initial_price, offer_price, 0, numbers[0]]]
                try:
                    best_shops = find_best_shops(shop_data, sample_input)
                    delivery_time = best_shops['Delivery_Time'].values[0]
                    book_price = numbers[0]
                    shop_name = best_shops['Shop_Name'].values[0]
                except Exception as e:
                    print(f"Error: {e}")

            if numbers:  
                try:
                    response = determine_response(user_input, best_shops)
                    print("Bot: ", response)
                    context += f"\nUser: {user_input}\nAI: {response}"

                    if "Do you accept?" in response:
                        while True:  # Loop to validate the response
                            next_input = input("You: ").lower()
                            
                            # Check if user accepts the deal
                            if next_input in affirmative_responses:
                                print(f"Deal accepted! Shop: {shop_name}, Delivery Time: {delivery_time} days.")
                                print("Thank you for using DealAI! It was a great transaction")
                                break  # End the conversation
                            elif next_input in rejection_responses:
                                print("Bot: Let's try again :)")
                                break  # Restart the negotiation or offer
                            else:
                                print("Bot: Hmm, do you accept or reject my previous offer?")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Bot: Are you taking the deal?")
        else:
            print("Bot: Let's focus :)")
            context += f"\nUser: {user_input}\nAI: Let's focus :)"