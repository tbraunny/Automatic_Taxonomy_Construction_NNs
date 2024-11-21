import requests
from llama_index.core import Document, VectorStoreIndex, Settings

'''
Example usage:


from utils.query_rag import LocalDocumentIndexer

pdf_path = "./data/papers/AlexNet.pdf"

documents = load_pdf(pdf_path)
documents = chunk_document(documents)
embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
llm_model = LLMModel(model_name="llama3.2:1b").get_llm()
rag_query_engine = LocalDocumentIndexer(embed_model=embed_model, llm_model=llm_model, documents=documents).get_rag_query_engine()

response = rag_query_engine.query("What is this paper about!")

'''

class LocalDocumentIndexer:
    """
    A utility class for creating and managing a document index locally.
    """
    def __init__(self, documents=None, embed_model=None, llm_model=None):
        if embed_model is None or llm_model is None or documents is None:
            raise ValueError("embed_model llm_model, documents must be provided.")
        
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.vector_index = None

        Settings.embed_model = self.embed_model
        Settings.llm = llm_model

        self.create_index(documents)

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

    def get_rag_query_engine(self):
        """
        Returns a query engine for the local index.

        Can call .query('') on this object
        """
        if self.vector_index is None:
            raise ValueError("Local vector index has not been created.")
        return self.vector_index.as_query_engine()
    
'''
Example Usage:

from utils.remote_rag import RemoteDocumentIndexer

device_ip="100.105.5.55"
port=5000
query_engine = RemoteDocumentIndexer(device_ip,port).get_rag_query_engine()
response = query_engine.query("What is this paper about")

By default:
Pdf points to AlexNet in ./data/papers/AlexNet.pdf
LLM model is llama3.1:8b
Embed Model is all-MiniLM-L6-v2
'''

class RemoteDocumentIndexer:
    """
    A utility class for querying a remote document index.
    """
    def __init__(self, device_ip, port, llm_model_name=None, embed_model_name =None, pdf_path = None):
        if not device_ip or not port:
            raise ValueError("device_ip and port must be provided.")
        
        self.device_ip = device_ip
        self.port = port
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.pdf_path = pdf_path

    # Not set up yet!!!!!
    def index_documents(self, documents):
        """
        Sends documents to the remote server for indexing.
        """
        url = f"http://{self.device_ip}:{self.port}/api/index_documents"
        payload = {
            "model": self.llm_model_name,
            "documents": [doc.text for doc in documents]
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("Documents indexed successfully on the remote server.")
        except requests.exceptions.RequestException as e:
            print(f"Error indexing documents on remote server: {e}")

    def query(self, user_query):
        """
        Sends a query to the remote instance and retrieves the response.
        """
        url = f"http://{self.device_ip}:{self.port}/api/query"
        payload = {
            "llm_model_name": self.llm_model_name,
            "embed_model_name": self.embed_model_name,
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

    def get_rag_query_engine(self):
        """
        Returns the query engine interface for remote usage.
        """
        return self
