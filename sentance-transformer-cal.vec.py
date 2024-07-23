from sentence_transformers import SentenceTransformer
import os
import pickle

# Global variable for the root path
ROOT_PATH = "E:/conda/fasttext"

# Set up proxy and authentication if needed
os.environ['HTTP_PROXY'] = "http://192.168.8.21:7890"
os.environ['HTTPS_PROXY'] = "http://192.168.8.21:7890"

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')

# Define paths using the ROOT_PATH
data_dir = os.path.join(ROOT_PATH, "data", "fao780")
output_file = os.path.join(ROOT_PATH, "model", "indexed_document.pkl")

def read_text_files(data_dir):
    """Read text from .txt files."""
    documents = []
    filenames = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(data_dir, filename)
            
            # Try reading the file with utf-8, then fall back to latin-1 if necessary
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                    document = txt_file.read().strip()
            except UnicodeDecodeError:
                try:
                    # Fallback to latin-1 encoding
                    with open(txt_file_path, 'r', encoding='latin-1') as txt_file:
                        document = txt_file.read().strip()
                except Exception as e:
                    print(f"Error reading {txt_file_path}: {e}")
                    continue  # Skip files that cannot be read
                
            documents.append(document)
            filenames.append(filename)
    
    return documents, filenames

def save_vectors(documents, filenames, model, output_file):
    """Save document vectors and filenames to a pickle file."""
    print("Encoding documents...")
    document_vectors = model.encode(documents, convert_to_tensor=True).cpu().numpy()
    
    # Create a dictionary to store vectors and filenames
    indexed_documents = {
        "filenames": filenames,
        "vectors": document_vectors
    }
    
    # Save the dictionary as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(indexed_documents, f)

    print(f"Indexed documents saved to {output_file}")

if __name__ == '__main__':
    # Read documents and filenames
    documents, filenames = read_text_files(data_dir)
    
    # Save vectors and filenames
    save_vectors(documents, filenames, model, output_file)
