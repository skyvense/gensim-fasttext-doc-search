import os
import numpy as np
from gensim.models import FastText
from scipy.spatial.distance import cosine

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"

# Load the trained FastText model
def load_model(model_path):
    return FastText.load(model_path)

# Function to get average vector for a text
def get_text_vector(model, text):
    tokens = text.strip().split()
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Function to classify text using label vectors and get top N labels
def classify_text(model, text, labels, top_n=5):
    text_vector = get_text_vector(model, text)
    similarities = {}
    for label in labels:
        label_text = f'__label__{label}'  # Format the label like in training
        label_vector = get_text_vector(model, label_text)
        # Compute similarity between text vector and label vector
        similarity = 1 - cosine(text_vector, label_vector)
        similarities[label] = similarity
    
    # Sort labels by similarity score in descending order and get top N
    sorted_labels = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_labels = sorted_labels[:top_n]
    
    return top_labels

# Main function for interactive querying
def main():
    model_path = os.path.join(ROOT_PATH, "model/indexed.model")
    model = load_model(model_path)

    # Extract filenames from the data directory to use as labels
    data_directory = os.path.join(ROOT_PATH, "data/fao780")
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
