import os
import re 
import numpy as np
from gensim.models import FastText
from scipy.spatial.distance import cosine

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"

def load_model(model_path):
    print(f"Loading model from {model_path}")
    return FastText.load(model_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_text_vector(model, text):
    tokens = preprocess_text(text).strip().split()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        print("Warning: No tokens found in model vocabulary.")
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def classify_text(model, text, labels, top_n=15):
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

def main():
    model_path = os.path.join(ROOT_PATH, "model/indexed.model.krapivin2009")
    model = load_model(model_path)

    data_directory = os.path.join(ROOT_PATH, "data/krapivin2009")
    labels = [os.path.splitext(f)[0] for f in os.listdir(data_directory) if f.endswith('.txt')]

    print("FastText Model Loaded. You can now enter text to classify.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            print("Exiting...")
            break

        top_labels = classify_text(model, query, labels, top_n=5)
        print(f"Top Labels:")
        for label, similarity in top_labels:
            print(f"Label: {label}, Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
