import ollama
import chromadb

from langchain_core.documents.base import Document
from utils.document_json_utils import load_documents_from_json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document_preserving_header(documents, chunk_size=1000, chunk_overlap=200) -> list:
    """
    Splits a list of documents into smaller chunks while preserving metadata headers.

    :param documents: A list of Document objects to be split.
    :param chunk_size: The size of each chunk. Defaults to 1000.
    :param chunk_overlap: The overlap between consecutive chunks. Defaults to 200.
    :return: A list of chunked Document objects with metadata headers.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []
    
    for doc in documents:
        original_metadata = doc.metadata.copy()  # Ensure we retain metadata
        header = original_metadata.get("section_header", "No Header")  # Default if no header

        split_chunks = text_splitter.split_text(doc.page_content)

        for chunk_text in split_chunks:
            chunked_doc = Document(
                page_content=chunk_text,
                metadata={"section_header": header}  # Attach the original header
            )
            chunked_docs.append(chunked_doc)

    return chunked_docs

def embed_and_store_chunks(documents, collection):
    """
    Embed and store chunks in the vector database.

    :param documents: A list of chunked Document objects.
    :param collection: The vector database collection to store embeddings.
    """
    for i, doc in enumerate(documents):
        try:
            response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=doc.page_content)
            embedding = response.get("embedding")
            if embedding:
                doc_id = f"chunk_{i}"
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata]  # Include metadata
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
        response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=prompt)
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
    Ensures that each chunk retains its section header for better comprehension.

    :param prompt: The user query to answer.
    :param context: The retrieved document context.
    :return: A generated response string.
    """
    try:
        full_context = "\n\n".join([
            f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
            for chunk in context
        ])
        
        print("### Context Provided to LLM ###")
        print(full_context)
        print("################################")
        
        response = ollama.generate(
            model="deepseek-r1:32b",
            prompt=(
                f"Using this context:\n{full_context}\n\n"
                f"Answer the following question concisely and accurately:\n{prompt}"
            ),
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

# Process JSON to list of Document Objects
docs = load_documents_from_json(json_file_path)

# Split documents into smaller chunks while preserving header
split_docs = chunk_document_preserving_header(docs)

# Embed and store documents
embed_and_store_chunks(split_docs, collection)

# Retrieve relevant chunks for a prompt
prompt = "Describe the methodology used in the research to reduce overfitting, including any data augmentation techniques or regularization strategies. Wrap your answers in triple back ticks ``` ```"
max_chunks = 10
token_budget = 1024
relevant_chunks = retrieve_relevant_chunks_within_token_budget(prompt, collection, max_chunks=max_chunks, token_budget=token_budget)

# Format and generate the final response
if relevant_chunks:
    final_response = generate_optimized_response(prompt, relevant_chunks)
    print("\nFinal Answer:\n", final_response)
else:
    print("No relevant documents found.")
