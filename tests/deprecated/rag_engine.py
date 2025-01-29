import requests
from llama_index.core import Document, VectorStoreIndex, Settings

from utils.pdf_loader import load_pdf
from utils.preprocess_pdf import preprocess_pdf
from tests.deprecated.llm_model import LLMModel
from tests.deprecated.hf_embeddings import OllamaEmbeddingModel, HuggingFaceEmbeddingModel
from utils.doc_chunker import chunk_document, chunk_document_for_nlm_LayoutPDFReader


'''
Example usage:

from utils.rag_engine import LocalRagEngine

pdf_path = "data/raw/AlexNet.pdf"

query_engine = LocalRagEngine(pdf_path=pdf_path).get_rag_query_engine()
response = query_engine.query("What is this paper about?")
'''

class LocalRagEngine:
    """
    A utility class for creating and managing a document index locally.
    """
    def __init__(self, pdf_path=None,llm_model='llama3.1:8b-instruct-fp16'):
        if pdf_path is None:
            raise ValueError("PDF path must be provided.")
        
        documents = self.load_and_preprocess_pdf(pdf_path)

        # Initialize models
        self.embed_model = OllamaEmbeddingModel().get_model()
        self.llm_model = LLMModel(model_name=llm_model).get_llm()
        
        self.vector_index = None

        # Required settings for making the index locally run and not defaulted to openai
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm_model

        self.create_index(documents)

    @staticmethod
    def load_and_preprocess_pdf(pdf_path):
        """
        Loads and preprocesses the PDF content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of processed and chunked documents.
        """
        # New PDF loader that separates reading and preprocessing
        # docs = preprocess_pdf(pdf_path)
        # chunked_documents = chunk_document_for_nlm_LayoutPDFReader(docs)

        
        docs = load_pdf(pdf_path)
        chunked_documents = chunk_document(docs)

        return chunked_documents

    def get_relevant_chunks(self, prompt):
        """
        Retrieve raw relevant chunks for a given prompt without LLM processing.

        Args:
            prompt: The query string to retrieve relevant chunks.

        Returns:
            list: A list of strings, each representing a retrieved chunk of text.
        """
        if not self.vector_index:
            raise ValueError("Vector index has not been created.")
        
        retriever = self.vector_index.as_retriever()

        # Use the retriever to fetch relevant nodes
        retrieved_nodes = retriever.retrieve(prompt)

        # Extract text content from the retrieved nodes
        retrieved_texts = [node.get_content() for node in retrieved_nodes]
        combined_text = " ".join(retrieved_texts)

        return combined_text


    def create_index(self, documents):
        """
        Creates a VectorStoreIndex from a list of documents locally.
        """
        index_documents = [Document(text=doc.page_content) for doc in documents]

        self.vector_index = VectorStoreIndex.from_documents(
            index_documents,
            embed_model=self.embed_model,
            llm_predictor=self.llm_model
        )

    def get_rag_engine(self):
        """
        Returns a query engine for the local index.

        Can call .query('') on this object
        """
        if self.vector_index is None:
            raise ValueError("Local vector index has not been created.")
        return self.vector_index.as_query_engine()
    

'''
Example Usage:

from utils.rag_engine import RemoteRagEngine

device_ip="100.105.5.55"
port=5000
pdf_path='data/raw/AlexNet.pdf'
query_engine = RemoteRagEngine(device_ip=device_ip,port=port,pdf_path=pdf_path).get_rag_query_engine()
response = query_engine.query("What is this paper about")

By default:
Pdf points to AlexNet in ./data/raw/AlexNet.pdf
LLM model is llama3.1:8b
Embed Model is all-MiniLM-L6-v2
'''

class RemoteRagEngine:
    """
    A utility class for querying a remote document index.
    """
    def __init__(self, device_ip=None, port=None, pdf_path = None):
        if not device_ip or not port or not pdf_path:
            raise ValueError("device_ip, port, and pdf path must be provided.")
        
        self.device_ip = device_ip
        self.port = port
        self.pdf_path = pdf_path

    def query(self, user_query):
        """
        Sends a query to the remote instance and retrieves the response.
        """
        url = f"http://{self.device_ip}:{self.port}/api/query"
        payload = {
            "pdf_path": self.pdf_path,
            "query": user_query,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            generated_text = response_data.get('response', 'No response field found')
            return generated_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying remote model: {e}")
            return {
                'response': 'Error querying remote model'
            }

    def get_rag_engine(self):
        """
        Returns the query engine interface for remote usage.
        """
        return self
