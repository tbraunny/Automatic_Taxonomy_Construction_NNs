from llama_index.core import VectorStoreIndex, Document

'''
Example usage:
from utils.document_indexer import DocumentIndexer
indexer = DocumentIndexer(embed_model, llm_predictor)
vector_index = indexer.create_index(split_docs)
'''

class DocumentIndexer:
    """
    A utility class for creating and managing a document index.
    """
    def __init__(self, embed_model, llm_predictor):
        """
        Constructor for DocumentIndexer.
        :param embed_model: Embedding model object.
        :type embed_model: LangchainEmbedding
        :param llm_predictor: LLM predictor object.
        :type llm_predictor: LangChainLLM
        """
        self.embed_model = embed_model
        self.llm_predictor = llm_predictor
        self.vector_index = None

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
            llm_predictor=self.llm_predictor
        )
        print("VectorStoreIndex built.")
        return self.vector_index
