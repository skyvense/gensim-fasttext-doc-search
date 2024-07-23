import os
import pickle
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"
MODEL_PATH = os.path.join(ROOT_PATH, "model")
ENCODED_DATA_PATH = os.path.join(ROOT_PATH, "encoded_texts_krapivin2009.pkl")
KEY_FILES_PATH = os.path.join(ROOT_PATH, "data/krapivin2009")  # Adjust this path if needed

# Load the SentenceTransformer model
def load_model():
    return SentenceTransformer(MODEL_PATH)

# Load previously encoded data
def load_encoded_data():
    if os.path.exists(ENCODED_DATA_PATH):
        with open(ENCODED_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# Load keywords from .key files
def load_keywords():
    keywords_dict = {}
    key_files = [f for f in os.listdir(KEY_FILES_PATH) if f.endswith(".key")]
    for filename in tqdm(key_files, desc="Loading key files", unit="file"):
        key_file_path = os.path.join(KEY_FILES_PATH, filename)
        txt_filename = filename.replace(".key", ".txt")
        keywords = []
        with open(key_file_path, 'r', encoding='latin-1') as key_file:
            keywords = key_file.read().splitlines()
        
        if keywords:
            keywords_dict[filename] = {
                'txt_filename': txt_filename,
                'keywords': keywords
            }
            print(f"Loaded {len(keywords)} keywords from {filename} (Associated file: {txt_filename})")
    return keywords_dict

# Perform full-text search and return top N results
def fulltext_search(query, encoded_texts, model, top_n=5):
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

# Evaluate recall for a single key file
def evaluate_recall(keywords, encoded_texts, model, associated_file, top_n=5):
    true_positives = 0
    
    for keyword in keywords:
        print(f"Searching for keyword: '{keyword}'")
        results = fulltext_search(keyword, encoded_texts, model, top_n)
        top_filenames = {filename for filename, _ in results}
        
        print(f"Top-N results for keyword '{keyword}': {top_filenames}")
        
        if associated_file in top_filenames:
            print(f"Keyword '{keyword}' found in top-N results. Associated file: {associated_file}")
            true_positives += 1
        else:
            print(f"Keyword '{keyword}' NOT found in top-N results. Associated file: {associated_file}")
    
    recall = true_positives / len(keywords)  # Average recall for all keywords in the file
    return recall

# Main function
if __name__ == "__main__":
    # Load the pre-trained model and encoded data
    model = load_model()
    encoded_texts = load_encoded_data()
    
    if encoded_texts is None:
        raise FileNotFoundError("Encoded data not found. Please encode the texts first.")
    
    # Load keywords from .key files
    print("Loading keywords...")
    keywords_dict = load_keywords()
    
    # Select random 10 key files for testing
    print("Selecting random key files...")
    random_keys = random.sample(list(keywords_dict.items()), 10)
    
    # Top-N values to test
    top_n_values = range(1, 100)
    recalls = {n: [] for n in top_n_values}
    
    # Evaluate recall for each top-N value
    print("Evaluating recall for different top-N values...")
    for top_n in top_n_values:
        print(f"Processing top-{top_n} results...")
        for key_file, data in tqdm(random_keys, desc=f"Top-{top_n} Recall Evaluation", unit="file"):
            associated_file = data['txt_filename']
            keywords = data['keywords']
            print(f"Evaluating keywords for file: {associated_file}")
            recall = evaluate_recall(keywords, encoded_texts, model, associated_file, top_n=top_n)
            recalls[top_n].append(recall)
    
    # Calculate average recall for each top-N value
    avg_recalls = {n: np.mean(recalls[n]) for n in top_n_values}
    
    # Print average recalls
    print("Average Recall:")
    for n in top_n_values:
        print(f"Top-{n}: {avg_recalls[n]:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(top_n_values, [avg_recalls[n] for n in top_n_values], marker='o')
    plt.xlabel('Top-N')
    plt.ylabel('Average Recall')
    plt.title('Recall vs. Top-N')
    plt.xticks(top_n_values)
    plt.grid(True)
    plt.show()
