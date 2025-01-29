import code_extractor
import json

import langchain

from utils.llm_model import OllamaLLMModel
from utils.parse_annetto_structure import *
from utils.owl import *

"""
Setup rag pipeline for multiple code files
Use the JSON files created from code_extractor as context within RAG
"""

class rag():
    def __init__():
        print("super")

def main():
    print("get started")

if __name__ == '__main__':
    print("try some shit")
    main()

# import os
# from transformers import LlamaTokenizer
# import numpy as np

# # Assuming you have the tokenizer for Llama, or any other model
# tokenizer = LlamaTokenizer.from_pretrained('path_to_llama_tokenizer')

# def chunk_code(file_path, max_tokens=4096):
#     """
#     Function to chunk code into smaller parts that fit within the token limit.
#     """
#     with open(file_path, 'r') as file:
#         code = file.read()

#     # Tokenize the entire code
#     tokens = tokenizer.encode(code)
    
#     # If tokens exceed the max tokens allowed, chunk them
#     chunks = []
#     while len(tokens) > max_tokens:
#         chunks.append(tokens[:max_tokens])
#         tokens = tokens[max_tokens:]

#     # Add the remaining tokens as the final chunk
#     if len(tokens) > 0:
#         chunks.append(tokens)

#     return chunks

# def search_relevant_chunks(query, code_chunks):
#     """
#     A simple search method to filter code chunks based on a query.
#     This could be expanded to use embeddings for semantic search.
#     """
#     relevant_chunks = []
#     for chunk in code_chunks:
#         # Check if query is present in any chunk (simple substring match for now)
#         if query.lower() in tokenizer.decode(chunk).lower():
#             relevant_chunks.append(chunk)
#     return relevant_chunks

# def main():
#     code_files = ['file1.py', 'file2.py']  # List your Python files here
#     query = "initialization"  # Your search/query term for finding relevant code

#     all_chunks = []
#     for file_path in code_files:
#         chunks = chunk_code(file_path)
#         all_chunks.extend(chunks)

#     # Search for the relevant chunks that match the query
#     relevant_chunks = search_relevant_chunks(query, all_chunks)

#     # If we have relevant chunks, process them further
#     if relevant_chunks:
#         # For simplicity, let's just join the relevant chunks and pass them to the model
#         context_input = tokenizer.decode(np.concatenate(relevant_chunks))  # Join all relevant code chunks
#         print("Model Input:", context_input)
#         # Here, you can pass this context_input to Llama for processing
#     else:
#         print("No relevant chunks found for the query.")

# if __name__ == "__main__":
#     main()
