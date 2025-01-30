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
            print(type(doc.metadata))
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
prompt = "Describe the methodology used in the research to reduce overfitting, including any data augmentation techniques or regularization strategies."
max_chunks = 10
token_budget = 1024
relevant_chunks = retrieve_relevant_chunks_within_token_budget(prompt, collection, max_chunks=max_chunks, token_budget=token_budget)

# Format and generate the final response
if relevant_chunks:
    final_response = generate_optimized_response(prompt, relevant_chunks)
    print("\nFinal Answer:\n", final_response)
else:
    print("No relevant documents found.")


# import ollama
# import chromadb
# from langchain_core.documents.base import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils.document_json_utils import load_documents_from_json

# class VectorDatabase:
#     """
#     Handles interactions with the vector database.
#     """
#     def __init__(self, collection_name="file_docs"):
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection(name=collection_name)

#     def add_document(self, doc_id, embedding, content, metadata):
#         """Stores the document embedding in the collection."""
#         self.collection.add(
#             ids=[doc_id],
#             embeddings=[embedding],
#             documents=[content],
#             metadatas=[metadata]
#         )
    
#     def query(self, embedding, max_results=10):
#         """Retrieves relevant documents based on the query embedding."""
#         return self.collection.query(
#             query_embeddings=[embedding],
#             n_results=max_results
#         )

# class DocumentProcessor:
#     """
#     Handles document loading and chunking while preserving metadata.
#     """
#     def __init__(self, chunk_size=1000, chunk_overlap=200):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap
#         )

#     def chunk_documents(self, documents):
#         """Splits documents into chunks while retaining metadata."""
#         chunked_docs = []
#         for doc in documents:
#             original_metadata = doc.metadata.copy()
#             header = original_metadata.get("section_header", "No Header")
#             split_chunks = self.text_splitter.split_text(doc.page_content)
            
#             for i, chunk_text in enumerate(split_chunks):
#                 chunked_doc = Document(
#                     page_content=chunk_text,
#                     metadata={"section_header": header}
#                 )
#                 chunked_docs.append(chunked_doc)
#         return chunked_docs

# class Embedder:
#     """
#     Handles embedding of document chunks and queries.
#     """
#     @staticmethod
#     def embed_text(text):
#         """Generates embeddings for a given text using Ollama."""
#         response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=text)
#         return response.get("embedding")

# class ChunkRetriever:
#     """
#     Retrieves relevant document chunks based on query and token budget.
#     """
#     def __init__(self, vector_db, token_budget=1024):
#         self.vector_db = vector_db
#         self.token_budget = token_budget
    
#     def retrieve_chunks(self, query, max_chunks=10):
#         """Fetches the most relevant document chunks within the token budget."""
#         embedding = Embedder.embed_text(query)
#         if not embedding:
#             print("Error: Failed to generate embedding for query.")
#             return []
        
#         results = self.vector_db.query(embedding, max_results=max_chunks)
#         if not all(isinstance(results.get(key, []), list) for key in ['documents', 'distances', 'metadatas']):
#             print("Unexpected response format from vector database.")
#             return []
        
#         all_docs = [
#             {"content": doc, "score": score, "metadata": meta}
#             for docs, scores, metas in zip(
#                 results.get('documents', []),
#                 results.get('distances', []),
#                 results.get('metadatas', [])
#             )
#             for doc, score, meta in zip(docs, scores, metas)
#         ]
        
#         sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
        
#         context = []
#         total_tokens = 0
#         for doc in sorted_docs:
#             doc_tokens = len(doc["content"].split())  # Rough token estimation
#             if total_tokens + doc_tokens <= self.token_budget:
#                 context.append(doc)
#                 total_tokens += doc_tokens
#             else:
#                 break
#         return context

# class LLMQueryEngine:
#     """
#     Generates responses using an LLM based on retrieved chunks.
#     """
#     @staticmethod
#     def generate_response(prompt, context):
#         """Generates a response using an LLM based on the provided context."""
#         full_context = "\n\n".join([
#             f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
#             for chunk in context
#         ])
        
#         print("### Context Provided to LLM ###")
#         print(full_context)
#         print("################################")
        
#         response = ollama.generate(
#             model="deepseek-r1:32b",
#             prompt=(
#                 f"Using this context:\n{full_context}\n\n"
#                 f"Answer the following question concisely and accurately:\n{prompt}"
#             ),
#         )
#         return response.get('response', "No response generated.")

# # -------- Pipeline Execution -------- #
# if __name__ == "__main__":
#     # Initialize components
#     vector_db = VectorDatabase()
#     doc_processor = DocumentProcessor()
#     retriever = ChunkRetriever(vector_db)
    
#     # Load documents from JSON
#     json_file_path = "data/alexnet/doc_alexnet.json"
#     docs = load_documents_from_json(json_file_path)
    
#     # Chunk documents
#     split_docs = doc_processor.chunk_documents(docs)
    
#     # Embed and store document chunks
#     for i, doc in enumerate(split_docs):
#         embedding = Embedder.embed_text(doc.page_content)
#         if embedding:
#             vector_db.add_document(f"chunk_{i}", embedding, doc.page_content, doc.metadata)
#         else:
#             print(f"Error embedding chunk {i}")
    
#     # Query processing
#     prompt = "Describe the methodology used in the research to reduce overfitting."
#     relevant_chunks = retriever.retrieve_chunks(prompt)
    
#     # Generate response
#     if relevant_chunks:
#         final_response = LLMQueryEngine.generate_response(prompt, relevant_chunks)
#         print("\nFinal Answer:\n", final_response)
#     else:
#         print("No relevant documents found.")

