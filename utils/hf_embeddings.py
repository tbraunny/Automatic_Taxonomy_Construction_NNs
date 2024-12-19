from transformers import AutoModel, AutoTokenizer
import torch
from llama_index.core.base.embeddings.base import BaseEmbedding


class HFEmbedding(BaseEmbedding):
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
