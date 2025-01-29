import code_extractor # for future consideration
import json
import os
import glob
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

"""
Setup rag pipeline for multiple code files
Use the JSON files created from code_extractor as context within RAG
"""

def load_json(path):
    json_files = [f for f in os.listdir(path)]

    model_data = []

    for f in json_files:
        with open(os.path.join(path , f) , 'r') as file:
            model_data.append(json.load(file))

    return model_data


def chunk_code(model_json , chunk_size=512, chunk_overlap=100):
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "def ", "class " , "__init__"]
    )
    
    return code_splitter.split_text(model_json)


def embed_store(collection , code):
    """
    Embed & store chunks within vector database (chromadb)
    """

    for idx , snippet in enumerate(code):
        try:
            metadata = {"filename" : snippet[0].get("filename" , "unknown")} # retrieve filename for added context
            
            for i , chunk in enumerate(snippet):
                collection.add( # add to chromadb, embeddings handled by add function
                    id = [f"{idx} : {i}"] , 
                    documents = chunk , 
                    metadatas=[metadata]
                )
        except Exception as e:
            print(f"Error embeddings chunk {idx}: {e}")


def retrieval(prompt , collection , max_chunks=10 , token_budget=1024): # token limit version
    """
    Retrieve the relevant chunks for a prompt given a token budget (good practice?)
    """
    
    try:
        response = collection.query(query_texts=[prompt] , n_results=max_chunks)

        if not all(isinstance(response.get(key, []), list) for key in ['documents', 'distances', 'metadatas']): # check for proper vector keys
            print("Vectors returned in unknown format: ")
            print(response.get(key , None) for key in ['documents', 'distances', 'metadatas'])
            
            return []

        all_data = [
                {"content": doc, "score": score, "metadata": meta}
                for docs, scores, metas in zip(
                    response.get('documents', []),
                    response.get('distances', []),
                    response.get('metadatas', [])
                )
                for doc, score, meta in zip(docs, scores, metas)
        ]

        sorted_data = sorted(all_data , key=lambda x: x["score"] , reverse=True)
        
        context = []
        token_count = 0

        for data in sorted_data:
            data_tokens = len(data["content"].split()) # approx token count for data obj

            if (data_tokens + token_count >= token_budget): # check tokens used against budget
                break

            context.append(data)
            token_count += data_tokens

        return "\n".join(context)
    
    except Exception as e:
        print(f"Error retrieving relevant code snippets: {e}")
        return None
    

def retrieval_top_k(prompt , collection , top_k=5): # nearest k relevant chunks approach, much simpler
    """
    Retrieve relevant chunks within k neighbors for a given prompt
    """

    try:
        response = collection.query(query_texts=[prompt] , n_results=top_k)

        results = []
        for code , meta in zip(response['documents'][0], response['metadatas'][0]): # iterate over code & metadata
            results.append(f"File: {meta['filename']}\nCode:\n{code}\n\n")
        return "\n".join(results)
    except Exception as e:
        print(f"Error retrieving relevant code snippets: {e}")
        return None

def generate_response(prompt , context , model):
    """
    Optimize a response based upon relevant context retrieved
    """

    try:
        print("Context provided to the LLM: ")
        print(f"\t{context}")
        print("--------------------------------------------\n")

        response = ollama.generate(
            model = model , 
            prompt = (f"Using the following context: \n{context}\n\n"
                      f"Answer the following question concisely & accurately: \n{prompt}") ,
        )
        return response.get('response' , "No response generated (oopsie)")

    except Exception as e:
        print(f"Response failed to generate: {e}")
        return "Error in response generation (see terminal for details)."

def main():
    ann_name = "alexnet"
    path = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.json")

    model_data = load_json(path)

    split_code = chunk_code(model_data)

    # initialize chromadb & collection
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(
        name = "code_snippets",
        embedding_function = embedding_functions.OllamaEmbeddingFunction(model_name="mxbai-embed-large")
    )

    embed_store(chroma_collection , split_code)

    prompt = "Describe how the architecture of the model, specifically the forward pass, mitigates ovefitting within the model"
    
    # hyperparameters, test different values here (recommended values hardcoded in functions)
        # or ust go with retrieval_top_k i really dont give a shit
    max_chunks = 10
    token_budget = 1024
    model = "deepseek-r1:32b"

    relevant_chunks = retrieval(prompt , chroma_collection , max_chunks=max_chunks , token_budget=token_budget)

    if relevant_chunks:
        response = generate_response(prompt , relevant_chunks , model)
        print("\nFinal Response:\n" , response)
    else:
        print("No relevant code found for the requested model.")

if __name__ == '__main__':
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
