

'''
Example usage:

from utils.embedding_model import EmbeddingModel
embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
'''

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

class EmbeddingModel:
    """
    A utility class for initializing and retrieving an embedding model.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Constructor for embedding model.
        :param model_name: Name of the HuggingFace embedding model.
        :type model_name: string
        """
        self.embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=model_name)
        )

    def get_model(self):
        """
        Retrieves the embedding model.
        :return: The embedding model object.
        :rtype: LangchainEmbedding
        """
        return self.embed_model