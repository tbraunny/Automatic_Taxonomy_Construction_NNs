from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

'''
Example usage:
from utils.embedding_model import EmbeddingModel
embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
'''

MODEL_NAME = 'all-MiniLM-L6-v2'

class EmbeddingModel:
    """
    A utility class for initializing and retrieving an embedding model.
    """
    def __init__(self, model_name=MODEL_NAME):
        """
        Constructor for embedding model.
        :param model_name: Name of the HuggingFace embedding model.
        :type model_name: string
        """
        print("Initializing the embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.embed_model = LangchainEmbedding(self.embedding_model)
        print("Embedding model initialized.")

    def get_model(self):
        """
        Retrieves the embedding model.
        :return: The embedding model object.
        :rtype: LangchainEmbedding
        """
        return self.embed_model