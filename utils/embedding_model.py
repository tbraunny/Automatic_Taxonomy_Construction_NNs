

'''
Example usage:

from utils.embedding_model import EmbeddingModel
embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
'''

from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddingModel:
    """
    A utility class for initializing and retrieving an embedding model.
    """
    def __init__(self, model_name='nomic-embed-text'):
        """
        Constructor for embedding model.
        :param model_name: Name of the Ollama embedding model.
        :type model_name: string
        """
        self.embed_model = OllamaEmbeddings(model=model_name)

    def get_model(self):
        """
        Retrieves the embedding model.
        :return: The embedding model object.
        :rtype: OllamaEmbeddings
        """
        return self.embed_model


from transformers import AutoModel, AutoTokenizer
import torch
from llama_index.core.base.embeddings.base import BaseEmbedding


class HuggingFaceEmbeddingModel(BaseEmbedding):
    """
    A utility class for initializing and retrieving embeddings using Hugging Face models.
    """

    def __init__(self, model_name='microsoft/codebert-base'):
        """
        Constructor for embedding model.
        :param model_name: Name of the HuggingFace model.
        :type model_name: string
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _get_query_embedding(self, query):
        """
        Generate embeddings for a query (text or code snippet).
        :param query: Input text or code snippet.
        :type query: string
        :return: Embedding vector as a list of floats.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the mean of the last hidden states to get a single vector
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.tolist()

    def get_text_embedding(self, text):
        """
        Wrapper method for getting text embeddings.
        :param text: Input text.
        :return: Embedding vector.
        """
        return self._get_query_embedding(text)
