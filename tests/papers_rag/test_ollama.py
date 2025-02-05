import ollama
import chromadb

from langchain_core.documents.base import Document
from utils.document_json_utils import load_documents_from_json
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.doc_chunker import semantically_chunk_documents

def embed_and_store_chunks(documents, collection):
    """
    Embed and store chunks in the vector database.

    :param documents: A list of chunked Document objects.
    :param collection: The vector database collection to store embeddings.
    """
    for i, doc in enumerate(documents):
        try:
            response = ollama.embeddings(model="bge-m3:latest", prompt=doc.page_content)
            embedding = response.get("embedding")
            if embedding:
                doc_id = f"chunk_{i}"
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata]  # Note: metadata no longer contains header info.
                )
            else:
                raise ValueError(f"Embedding failed for chunk {i}")
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")

def retrieve_relevant_chunks_within_token_budget(prompt, collection, max_chunks=10, token_budget=1024):
    """
    Retrieve the most relevant chunks for the given prompt, adhering to a token budget.

    :param prompt: The query prompt.
    :param collection: The vector database collection to query from.
    :param max_chunks: Maximum number of chunks to retrieve.
    :param token_budget: Token limit for the retrieved chunks.
    :return: A list of relevant chunk dictionaries.
    """
    try:
        response = ollama.embeddings(model="bge-m3:latest", prompt=prompt)
        embedding = response.get("embedding")
        if embedding:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=max_chunks
            )
            if not all(isinstance(results.get(key, []), list) for key in ['documents', 'distances', 'metadatas']):
                print("Unexpected response format from vector database.")
                return []
            
            all_docs = [
                {"content": doc, "score": score, "metadata": meta}
                for docs, scores, metas in zip(
                    results.get('documents', []),
                    results.get('distances', []),
                    results.get('metadatas', [])
                )
                for doc, score, meta in zip(docs, scores, metas)
            ]
            
            sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
            
            context = []
            total_tokens = 0
            for doc in sorted_docs:
                doc_tokens = len(doc["content"].split())  # Rough token estimation
                if total_tokens + doc_tokens <= token_budget:
                    context.append(doc)
                    total_tokens += doc_tokens
                else:
                    break
            return context
        else:
            print("Error: Failed to generate embedding for prompt.")
            return []
    except Exception as e:
        print(f"Error querying relevant documents: {e}")
        return []
    
def generate_optimized_response(prompt, context):
    """
    Generate a concise and contextually relevant response using an LLM.
    Note: Since header information is no longer appended to the chunks, the header is not prepended here.
    
    :param prompt: The user query to answer.
    :param context: The retrieved document context.
    :return: A generated response string.
    """
    try:
        full_context = "\n\n".join([
            # Originally, each chunk was prefixed with its header:
            # f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
            chunk['content']
            for chunk in context
        ])
        
        print("### Context Provided to LLM ###")
        print(full_context)
        print("################################")

        instructions = (
            "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
            "Below are several context sections. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
            "### Evidence Blocks:\n"
            f"{full_context}\n\n"
            f"Query: {prompt}\n"
            "Do not abreviate answers."
            "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
            "with your answer listed as the value. For example, the last line of your output should be:\n"
            """{"name":"John"}"""
            """{"age":30}"""
            """{"cars":["Ford", "BMW", "Fiat"]}"""
        )
        
        response = ollama.generate(
            model="deepseek-r1:32b",
            prompt=instructions,
            options={"num_ctx": 1500},
            # An alternative prompt format (commented out):
            # prompt=(
            #     f"Using this context:\n{full_context}\n\n"
            #     f"Answer the following question concisely and accurately:\n{prompt}"
            # ),
        )
        return response.get('response', "No response generated.")
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error in response generation."

# Initialize ChromaDB client and create a collection
client = chromadb.Client()
collection = client.create_collection(name="file_docs")

# Path to JSON file containing doc object
json_file_path = "data/alexnet/doc_alexnet.json"
json_file_path = "data/resnet/doc_resnet.json"


# Process JSON to list of Document objects
docs = load_documents_from_json(json_file_path)

# Split documents into smaller chunks.
# (Note: semantically_chunk_documents is used here, so ensure it aligns with your intended behavior.)
split_docs = chunked_docs = semantically_chunk_documents(docs)

# Embed and store the document chunks.
embed_and_store_chunks(split_docs, collection)

# Retrieve relevant chunks for a prompt.
prompt = (
    "Give each layer of this neural network sequentially. Do not generalize internal layers and include all types of layers such as modifiction and activation layers"
    )
max_chunks = 30
token_budget = 2048
relevant_chunks = retrieve_relevant_chunks_within_token_budget(prompt, collection, max_chunks=max_chunks, token_budget=token_budget)

import time
# Generate the final response using the retrieved context.
if relevant_chunks:
    start_time = time.time()
    final_response = generate_optimized_response(prompt, relevant_chunks)
    print("\nFinal Answer:\n", final_response, "\nIn Time:", time.time() - start_time)
else:
    print("No relevant documents found.")