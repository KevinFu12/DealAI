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
import pandas as pd

# Initialize NLTK
nltk.download('punkt', quiet=True)

def detect_bargaining(input_text):
    tokens = word_tokenize(input_text.lower())
    bargaining_keywords = ["maybe", "buy", "sell", "offer", "price", "about", "dunno"]
    bargaining_verbs = ["buy", "sell"]

    # Only digits
    if len(tokens) == 1 and tokens[0].isdigit():
        return True

    for token in tokens:
        if token in bargaining_keywords:
            return True
        if token in bargaining_verbs and tokens[0] == "i":
            return True

    return False

warnings.filterwarnings("ignore")

# Ignore warnings
def ignore_warnings(*args, **kwargs):
    pass

check_array = ignore_warnings

# Import BargainModel
spec = importlib.util.spec_from_file_location("BargainModel", "BargainModel.py")
BargainModel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BargainModel)

def select_random_shop(shop_data):
    try:
        random_shop = shop_data.sample(n=1)
        return random_shop
    except Exception as e:
        print(f"Error selecting random shop: {e}")
        return None

# Find 3 best shops
def find_best_shops(shop_data, sample_input):
    try:
        return BargainModel.find_best_shop_knn(shop_data, sample_input, k=3)
    except Exception as e:
        print(f"Error finding best shops: {e}")
        return None

def determine_response(user_input, best_shops):
    if not isinstance(best_shops, pd.DataFrame):
        raise ValueError("best_shops must be a pandas DataFrame.")
    
    user_price = int(user_input.split()[-1])  # Ensure user input is valid
    best_shops['Price_Difference'] = abs(best_shops['Final_Price'] - user_price)
    closest_shop = best_shops.loc[best_shops['Price_Difference'].idxmin()]
    
    lowest = math.ceil(closest_shop['Final_Price'])
    highest = math.ceil(best_shops['Final_Price'].max())
    delivery_time = closest_shop['Delivery_Time']
    shop_name = closest_shop['Shop_Name']

    if user_price <= 0.8 * lowest:
        prompt = f"The offer is rejected, the highest I can do is {lowest}. Sorry!"
    elif user_price < lowest:
        prompt = f"Counteroffer: {math.ceil(1.01 * lowest)}, delivery in {delivery_time} days from {shop_name}."
    elif user_price >= lowest and user_price <= highest:
        prompt = f"Accept! Delivery in {delivery_time} days from {shop_name} for {user_price}. Do you accept?"
    else:
        prompt = f"OF COURSE, ACCEPT. Thank You! Delivery in {delivery_time} days from {shop_name} for {user_price}."

    return prompt

def extract_numbers(user_input):
    return [int(num) for num in re.findall(r'\d+', user_input)]

# Main program
if __name__ == "__main__":
    # Load shops data
    try:
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'shop_data.csv')
        shop_data = BargainModel.load_data(file_path)

        if not isinstance(shop_data, pd.DataFrame):
            raise ValueError("Shop data is not a DataFrame.")
    except Exception as e:
        print(f"Error loading shop data: {e}")
        exit()

    # Initialize
    template = """
    Answer the question below.

    Here is the conversation history: {context}
    Question: {question}
    Answer:
    """
    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    random_shop = select_random_shop(shop_data)
    if random_shop is None:
        exit()

    offer_price = math.ceil(random_shop['Selling_Price'].values[0])
    initial_price = random_shop['Initial_Price'].values[0]

    context = ""
    print("Welcome, I am DealAI, type 'exit' to quit")
    res = chain.invoke({"context": "", "question": f"You are DealAI, an e-commerce bargain bot. First, crack a dad joke or a pun. You are not allowed to talk about anything other than bargaining. Keep that in mind. First you introduce yourself. Explain that you have 1 interesting modern novel to offer with a price of around {offer_price} thousand rupiahs. Tell the user the title and the story."})
    print(res)

    best_shops = None

    affirmative_responses = {"deal", "yeah", "ye", "y", "yea", "ya", "yes", "i think so", "i do", "uh huh"}
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
                sample_input = [[initial_price, offer_price, 0, numbers[0]]] if numbers else []
                try:
                    best_shops = find_best_shops(shop_data, sample_input)
                    if best_shops is None:
                        raise ValueError("Could not find the best shops.")
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if numbers:
                try:
                    response = determine_response(user_input, best_shops)
                    print("Bot: ", response)
                    context += f"\nUser: {user_input}\nAI: {response}"

                    if "Do you accept?" in response:
                        while True:  # Validate response
                            next_input = input("You: ").lower()

                            if next_input in affirmative_responses:
                                print("Deal accepted!")
                                print("Thank you for using DealAI! It was a great transaction.")
                                break
                            elif next_input in rejection_responses:
                                print("Bot: Let's try again :)")
                                break
                            else:
                                print("Bot: Hmm, do you accept or reject my previous offer?")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Bot: Are you taking the deal?")
        else:
            print("Bot: Let's focus :)")
            context += f"\nUser: {user_input}\nAI: Let's focus :)"
