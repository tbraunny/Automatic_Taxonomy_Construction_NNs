import ollama
import chromadb
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_json_utils import load_documents_from_json
from sentence_transformers import CrossEncoder

class VectorDatabase:
    """
    Handles interactions with the vector database.
    """
    def __init__(self, collection_name="file_docs"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    def add_document(self, doc_id, embedding, content, metadata):
        """Stores the document embedding in the collection."""
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    def query(self, embedding, max_results=10):
        """Retrieves relevant documents based on the query embedding."""
        
        if not embedding or not isinstance(embedding, list):
            print("Error: Received empty or invalid embedding for query.")
            return {"documents": [], "distances": [], "metadatas": []}

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=max_results
        )

class DocumentProcessor:
    """
    Handles document loading and chunking while preserving metadata.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk_documents(self, documents):
        """Splits documents into chunks while retaining metadata."""
        chunked_docs = []
        for doc in documents:
            original_metadata = doc.metadata.copy()
            header = original_metadata.get("section_header", "No Header")
            split_chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(split_chunks):
                chunked_doc = Document(
                    page_content=chunk_text,
                    metadata={"section_header": header}
                )
                chunked_docs.append(chunked_doc)
        return chunked_docs

class Embedder:
    """
    Handles embedding of document chunks and queries.
    """
    @staticmethod
    def embed_text(text):
        """Generates embeddings for a given text using Ollama."""
        
        # Ensure text is not empty
        if not text or not text.strip():
            print("Error: Empty text passed to embedding function.")
            return None
        
        try:
            response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=text)
            
            # Check if response is valid
            if not response or "embedding" not in response:
                print(f"Embedding API call failed for text: {text}")
                return None
            
            embedding = response.get("embedding", None)
            if not embedding:
                print(f"Empty embedding received for: {text}")
                return None
            
            return embedding
        except Exception as e:
            print(f"Exception during embedding: {str(e)}")
            return None


class ChunkRetriever:
    """
    Implements advanced retrieval strategies for ChatGPT-like responses.
    """
    def __init__(self, vector_db, token_budget=1024, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_db = vector_db
        self.token_budget = token_budget
        self.reranker = CrossEncoder(reranker_model)
    
    def query_expansion(self, query):
        """Expands queries using LLM-based rewriting."""
        if not query or not query.strip():
            print("Error: Empty query received for expansion.")
            return [query]

        try:
            response = ollama.generate(
                model="llama3.1:8b-instruct-fp16",
                prompt=f"Generate a set of alternative phrasings and synonyms for this query: {query}"
            )
            expanded_queries = response.get('response', '').split('\\n')

            # Remove empty results
            expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
            
            if not expanded_queries:
                print("⚠️ No expansions generated. Using original query.")
                return [query]

            return [query] + expanded_queries[:3]  # Limit to 3 expansions
        except Exception as e:
            print(f"Exception in query expansion: {str(e)}")
            return [query]

    
    def retrieve_chunks(self, query, max_chunks=10):
        """Advanced retrieval pipeline with hybrid search, MMR, and re-ranking."""
        expanded_queries = self.query_expansion(query)
        embeddings = [Embedder.embed_text(q) for q in expanded_queries]

        # Ensure embeddings are valid
        embeddings = [e for e in embeddings if e is not None]
        if not embeddings:
            print("Error: No valid embeddings generated.")
            return []
        
        results = []
        for emb in embeddings:
            res = self.vector_db.query(emb, max_results=max_chunks)
            if res and 'documents' in res and 'metadatas' in res:
                docs = res.get('documents', [])
                metas = res.get('metadatas', [])
                results.extend(zip(docs, metas))

        # Ensure unique results and fix TypeError by converting to tuples
        unique_results = list({doc: doc for doc, _ in results}.values())

        if not unique_results:
            print("⚠️ No relevant results retrieved from vector database.")
            return []

        # Ensure correct return format (list of dictionaries)
        formatted_results = [{"content": doc, "metadata": meta} for doc, meta in results]

        return formatted_results


class LLMQueryEngine:
    """
    Generates responses using an LLM based on retrieved chunks.
    """
    @staticmethod
    def generate_response(prompt, context):
        """Generates a response using an LLM based on the provided context."""

        if not isinstance(context, list) or not all(isinstance(chunk, dict) and "content" in chunk for chunk in context):
            print("Error: Retrieved chunks are not in the expected dictionary format.")
            print(f"Received context: {context}")
            return "Error: Invalid document format."

        full_context = "\n\n".join([f"{chunk['content']}" for chunk in context])

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




if __name__ == "__main__":
    vector_db = VectorDatabase()
    doc_processor = DocumentProcessor()
    retriever = ChunkRetriever(vector_db)
    
    json_file_path = "data/alexnet/doc_alexnet.json"
    docs = load_documents_from_json(json_file_path)
    
    split_docs = doc_processor.chunk_documents(docs)
    
    for i, doc in enumerate(split_docs):
        embedding = Embedder.embed_text(doc.page_content)
        if embedding:
            vector_db.add_document(f"chunk_{i}", embedding, doc.page_content, doc.metadata)
        else:
            print(f"Error embedding chunk {i}")
    
    prompt = "Describe the methodology used in the research to reduce overfitting."
    relevant_chunks = retriever.retrieve_chunks(prompt)
    
    if relevant_chunks:
        final_response = LLMQueryEngine.generate_response(prompt, relevant_chunks)
        print("\nFinal Answer:\n", final_response)
    else:
        print("No relevant documents found.")
