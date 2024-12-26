#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\offerBot\Scripts\Activate.ps1

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import importlib.util
import os
import math
import re
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
import warnings
import nltk
from nltk.tokenize import word_tokenize

def detect_bargaining(input_text):
    tokens = word_tokenize(input_text.lower())
    bargaining_keywords = ["", "maybe", "buy", "sell", "offer", "price", "about", "dunno"]
    bargaining_verbs = ["buy", "sell"]

    # Periksa apakah input hanya mengandung angka
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

# Membuat scaler
scaler = StandardScaler()

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
    lowest = math.ceil(min(best_shops['Final_Price']))
    highest = math.ceil(max(best_shops['Final_Price']))
    delivery_time = best_shops['Delivery_Time'].values[0]
    shop_name = best_shops['Shop_Name'].values[0]

    user_price = int(user_input.split()[-1])
    if user_price <= 0.8 * lowest:
        # Reject
        prompt = f"The offer is rejected, the highest i can do is {lowest}. Sorry!"
    elif user_price < lowest:
        # Counteroffer
        prompt = f"Counteroffer: {math.ceil(1.01 * lowest)}, I think this is more competitive. Delivery in {delivery_time} days."
    elif user_price >= lowest and user_price <= highest:
        # Accept
        prompt = f"Accept! Your book will arrive in {delivery_time} days for {user_price}. Do you accept?"
    elif user_price > highest:
        prompt = f"OF COURSE, ACCEPT. Thank You! Delivery in {delivery_time} days for {user_price}. Do you accept?"

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
    res = chain.invoke({"context": "", "question": f"You are DealAI, an e-commerce bargain bot. You are not allowed to talk about anything other than bargaining. Keep that in mind. First you introduce yourself. Explain that you have 1 famous novel to offer with price of {offer_price}, tell the user the title and the story"})
    print(res)

    best_shops = None

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Thank you for using DealAI, have a nice day.")
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
                        next_input = input("You: ")
                        if next_input.lower() == "deal":
                            print(f"Deal accepted! Shop: {shop_name}, Price: {book_price}, Delivery Time: {delivery_time} days.")
                            print("Thank you for using DealAI!. It was a great transaction")
                            break
                        else:
                            print("Bot: Let's try again :)")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Bot: Are you taking the deal?")
        else:
            print("Bot: Let's focus :)")
            context += f"\nUser: {user_input}\nAI: Let's focus :)"