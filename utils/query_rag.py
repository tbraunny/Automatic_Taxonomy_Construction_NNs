'''
Example usage:
from utils.document_indexer import DocumentIndexer

query_engine = DocumentIndexer(embed_model, llm_model,split_docs).get_rag_query_engine()
user_query='what is this about'
response = query_engine.query(user_query)
'''

from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings


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
    
    def get_rag_query_engine(self):
        return self.vector_index.as_query_engine()
