

'''
Example usage:

from utils.embedding_model import EmbeddingModel
embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
'''

from langchain_ollama import OllamaEmbeddings

class EmbeddingModel:
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
    

from transformers import AutoModel, AutoTokenizer
import torch

class CodeBERTEmbeddingModel:
    """
    A utility class for initializing and retrieving embeddings using CodeBERT.
    """
    def __init__(self, model_name='microsoft/codebert-base'):
        """
        Constructor for CodeBERT embedding model.
        :param model_name: Name of the HuggingFace CodeBERT model.
        :type model_name: string
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text):
        """
        Generate embeddings for a given text or code snippet.
        :param text: Input text or code snippet.
        :type text: string
        :return: Embedding vector.
        :rtype: torch.Tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the mean of the last hidden states to get a single vector
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


def test_codebert_embedding():
    # Initialize the CodeBERT embedding model
    embedding_model = CodeBERTEmbeddingModel()

    # Test data
    test_text = "What is the capital of France?"
    test_code = """
    def hello_world():
        print('Hello, world!')
    """

    # Generate embeddings
    text_embedding = embedding_model.embed_text(test_text)
    code_embedding = embedding_model.embed_text(test_code)

    # Print results
    print("Text Embedding (Shape: {}):".format(text_embedding.shape))
    print(text_embedding)
    print("\nCode Embedding (Shape: {}):".format(code_embedding.shape))
    print(code_embedding)


test_codebert_embedding()

