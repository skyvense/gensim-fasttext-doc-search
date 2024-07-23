import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Set up proxy and authentication if needed
os.environ['HTTP_PROXY'] = "http://192.168.8.21:7890"
os.environ['HTTPS_PROXY'] = "http://192.168.8.21:7890"

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"
MODEL_PATH = os.path.join(ROOT_PATH, "model")
DATA_PATH = os.path.join(ROOT_PATH, "data/krapivin2009")
ENCODED_DATA_PATH = os.path.join(ROOT_PATH, "encoded_texts_krapivin2009.pkl")

# Function to load full text from .txt files with progress bar
def load_texts():
    texts_dict = {}
    txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
    for filename in tqdm(txt_files, desc="Loading texts", unit="file"):
        txt_file_path = os.path.join(DATA_PATH, filename)
        with open(txt_file_path, 'r', encoding='latin-1') as txt_file:
            full_text = txt_file.read()
            texts_dict[filename] = full_text
    return texts_dict

# Function to create or load SentenceTransformer model
def load_or_create_model():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
        model.save(model_path)
    else:
        model = SentenceTransformer(model_path)
    return model

# Function to encode full texts using SentenceTransformer with progress bar
def encode_texts(model, texts_dict):
    encoded_texts = {}
    for filename, text in tqdm(texts_dict.items(), desc="Encoding texts", unit="file"):
        text_embedding = model.encode([text])[0]  # Encode and get the first (and only) embedding
        encoded_texts[filename] = text_embedding
    return encoded_texts

# Function to save encoded data to a file
def save_encoded_data(encoded_texts):
    with open(ENCODED_DATA_PATH, 'wb') as f:
        pickle.dump(encoded_texts, f)

# Function to load encoded data from a file
def load_encoded_data():
    if os.path.exists(ENCODED_DATA_PATH):
        with open(ENCODED_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# Function to perform fulltext search and return top N results
def fulltext_search(query, encoded_texts, top_n=5):
    query_embedding = model.encode([query])[0]  # Encode the query

    # Calculate similarity scores for all documents
    scores = []
    for filename, text_embedding in encoded_texts.items():
        score = cosine_similarity([query_embedding], [text_embedding])[0][0]
        scores.append((filename, score))
    
    # Sort results by score in descending order
    sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Return top N results
    top_results = sorted_results[:top_n]
    return top_results

# Main function
if __name__ == "__main__":
    # Step 1: Create or load SentenceTransformer model
    model = load_or_create_model()
    
    # Step 2: Load encoded data if available, otherwise encode texts
    encoded_texts = load_encoded_data()
    
    if encoded_texts is None:
        print("No pre-encoded data found. Encoding texts...")
        # Load full text from .txt files
        texts_dict = load_texts()
        
        # Encode full texts into vectors
        encoded_texts = encode_texts(model, texts_dict)
        
        # Save encoded data for future use
        save_encoded_data(encoded_texts)
    else:
        print("Loaded pre-encoded data.")
    
    # Step 3: Interactive querying loop
    while True:
        keyword_query = input("Enter a keyword to search (or type 'exit' to quit): ")
        if keyword_query.lower() == 'exit':
            print("Exiting...")
            break
        
        top_results = fulltext_search(keyword_query, encoded_texts)
        
        if top_results:
            print(f"Top matches for '{keyword_query}':")
            for idx, (filename, score) in enumerate(top_results, start=1):
                print(f"{idx}. {filename} (Score: {score:.4f})")
        else:
            print(f"No match found for '{keyword_query}'")
