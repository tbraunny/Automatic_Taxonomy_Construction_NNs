import sys
import re
from transformers import AutoTokenizer

def read_owl_file(file_path):
    """
    Read the contents of an OWL file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    file_path = """/home/richw/richie/Automatic_Taxonomy_Construction_NNs/data/owl/annett-o-0.1.owl"""
    text = read_owl_file(file_path)

    # Use a simple tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Tokenize the text and count tokens
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    
    print(f"Estimated token count: {token_count}")

if __name__ == "__main__":
    main()
