import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from gensim.models import FastText
from tqdm import tqdm

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"

# Function to read a single .txt file and prepend each line with the filename as the label
def process_file(filename, directory):
    label = os.path.splitext(filename)[0]  # Get the filename without extension
    labeled_texts = []
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
        for line in file:
            words = line.strip().split()  # Tokenize the line into words
            labeled_texts.append([f'__label__{label}'] + words)
    return labeled_texts

# Function to read .txt files using multithreading
def read_labeled_text_contents(directory):
    labeled_texts = []
    filenames = [f for f in os.listdir(directory) if f.endswith('.txt')]
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, filename, directory): filename for filename in filenames}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing Files"):
            labeled_texts.extend(future.result())
    return labeled_texts

# Function to train FastText model on the tokenized labeled text contents with progress tracking
def train_fasttext_model(labeled_texts, model_save_path):
    model = FastText(vector_size=100, window=5, min_count=1, sg=1, workers=20)
    model.build_vocab(labeled_texts)

    # Manually track training progress
    total_examples = len(labeled_texts)
    epochs = 10
    progress_bar = tqdm(total=epochs, desc="Training Epochs", unit="epoch")

    def train_callback(*args, **kwargs):
        progress_bar.update(1)

    # Train the model with a callback to update progress bar
    for epoch in range(epochs):
        model.train(labeled_texts, total_examples=total_examples, epochs=1)
        train_callback()
    
    progress_bar.close()
    model.save(model_save_path)

# Main function
def main():
    data_directory = os.path.join(ROOT_PATH, "data/fao780")
    model_save_path = os.path.join(ROOT_PATH, "model/indexed.model")

    labeled_texts = read_labeled_text_contents(data_directory)
    train_fasttext_model(labeled_texts, model_save_path)

if __name__ == "__main__":
    main()
