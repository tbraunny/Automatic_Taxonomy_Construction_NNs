'''
Example usage:
from utils.document_indexer import DocumentIndexer

query_engine = DocumentIndexer(embed_model, llm_model,split_docs).get_rag_query_engine()
user_query='what is this about'
response = query_engine.query(user_query)
'''

from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings

import requests


class DocumentIndexer:
    """
    A utility class for creating and managing a document index.
    """
    def __init__(self, embed_model, llm_model,documents):
        """
        Constructor for DocumentIndexer.
        :param embed_model: Embedding model object.
        :type embed_model: LangchainEmbedding
        :param llm_predictor: LLM predictor object.
        :type llm_predictor: LangChainLLM
        """
        self.embed_model = embed_model
        self.llm_model = llm_model
        # Set the global settings for LLM and embedding model
        Settings.llm = self.llm_model
        Settings.embed_model = self.embed_model
        self.vector_index = None
        self.create_index(documents)

    def create_index(self, documents):
        """
        Creates a VectorStoreIndex from a list of documents.
        :param documents: List of documents to index.
        :type documents: list
        :return: VectorStoreIndex object.
        :rtype: VectorStoreIndex
        """
        print("Creating LlamaIndex documents...")
        index_documents = [Document(text=doc.page_content) for doc in documents]
        print(f"Created {len(index_documents)} LlamaIndex documents.")

        print("Building the VectorStoreIndex...")
        self.vector_index = VectorStoreIndex.from_documents(
            index_documents, 
            embed_model=self.embed_model, 
            llm_predictor=self.llm_model
        )
        print("VectorStoreIndex built.")
        # return self.vector_index

    def get_rag_query_engine(self, remote=False, device_ip=None, port=None):
        """
        Returns a query engine for either the local index or a remote Ollama instance.
        :param remote: Whether to query a remote Ollama instance.
        :param device_ip: IP address of the device running Ollama (required if remote=True).
        :param port: Port number where Ollama is running (required if remote=True).
        :return: Query engine with a `.query()` interface.
        """
        if not remote:
            # Return the local query engine
            return self.vector_index.as_query_engine()
        else:
            # Validate remote parameters
            if not device_ip or not port:
                raise ValueError("For remote queries, device_ip and port must be provided.")
            # Return the remote query engine wrapped in a class
            return RemoteQueryEngine(device_ip, port)


class RemoteQueryEngine:
    """
    A wrapper for querying a remote Ollama instance with a `.query()` interface.
    """
    def __init__(self, device_ip, port):
        self.device_ip = device_ip
        self.port = port

    def query(self, user_query):
        """
        Sends a query to the remote Ollama instance.
        :param user_query: Query string to send to Ollama.
        :return: Response from Ollama.
        """
        url = f"http://{self.device_ip}:{self.port}/api/query"
        payload = {"query": user_query}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()  # Process the response as needed
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return None
    
    # def get_rag_query_engine(self):
    #     return self.vector_index.as_query_engine()