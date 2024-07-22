import os
import numpy as np
import re
import matplotlib.pyplot as plt
from gensim.models import FastText
from scipy.spatial.distance import cosine
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"
TOP_N = 100
SAMPLE_SIZE = 20  # Number of key files to randomly sample

def load_model(model_path):
    print(f"Loading model from {model_path}")
    return FastText.load(model_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def get_text_vector(model, text):
    tokens = preprocess_text(text).strip().split()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        print("Warning: No tokens found in model vocabulary.")
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def classify_text(model, text, labels, top_n=TOP_N):
    text_vector = get_text_vector(model, text)
    similarities = {}
    for label in labels:
        label_text = f'__label__{label}'
        label_vector = get_text_vector(model, label_text)
        similarity = 1 - cosine(text_vector, label_vector)
        similarities[label] = similarity
    
    sorted_labels = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_labels = sorted_labels[:top_n]
    
    return top_labels

def calculate_recall(true_labels, predicted_labels):
    true_positives = sum(1 for label in true_labels if label in predicted_labels)
    recall = true_positives / len(true_labels)
    return recall

def process_keyword(args):
    model, keyword, labels, filename = args
    keyword = keyword.strip()
    top_labels = classify_text(model, keyword, labels)
    predicted_labels = [label for label, _ in top_labels]
    recall = calculate_recall([filename], predicted_labels)
    return recall

def plot_recalls(recalls):
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, marker='o', linestyle='-', color='b')
    plt.title('Recall Rate per Keyword')
    plt.xlabel('Keyword Index')
    plt.ylabel('Recall Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('recall_vs_topn_fasttext.png')  # Save the figure
    plt.show()

def main():
    model_path = os.path.join(ROOT_PATH, "model/indexed.model")
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    data_directory = os.path.join(ROOT_PATH, "data/fao780")
    key_files = [f for f in os.listdir(data_directory) if f.endswith('.key')]
    labels = [os.path.splitext(f)[0] for f in os.listdir(data_directory) if f.endswith('.txt')]

    print(f"Found {len(key_files)} key files and {len(labels)} labels.")

    # Randomly sample 20 key files
    sampled_key_files = random.sample(key_files, min(SAMPLE_SIZE, len(key_files)))
    print(f"Randomly selected {len(sampled_key_files)} key files.")

    recalls = []

    tasks = []
    for key_file in sampled_key_files:
        filename = os.path.splitext(key_file)[0]
        key_file_path = os.path.join(data_directory, key_file)
        
        print(f"Processing key file: {key_file}")
        with open(key_file_path, 'r', encoding='latin-1', errors='ignore') as file:
            keywords = file.readlines()
        
        for keyword in keywords:
            tasks.append((model, keyword, labels, filename))
        print(f"Added {len(keywords)} keywords from {key_file} to task list.")

    print("Starting parallel processing of keywords...")
    with Pool(processes=cpu_count()) as pool:
        for recall in tqdm(pool.imap(process_keyword, tasks), total=len(tasks), desc="Processing keywords"):
            recalls.append(recall)

    average_recall = np.mean(recalls)
    print(f"Average Recall: {average_recall:.4f}")

    print("Plotting recall results...")
    plot_recalls(recalls)
    print("Recall plot saved as 'recall_vs_topn_fasttext.png'.")

if __name__ == "__main__":
    main()
