from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
Example Usage:

from utils.ollama_embeddings import find_relevant_content
relevant_content = find_relevant_content(query, text)
print("Most Relevant Content:", relevant_content)
"""

""" Needs to be rewritten to store text embedding for reuse!!!!!"""

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
    

def find_relevant_content(query, chunked_text, model_name='mxbai-embed-large', chunk_size=100):
    """
    Finds the most relevant content from the text based on the query using embeddings.
    :param query: The query string.
    :type query: string
    :param text: The chunked text to search within.
    :type text: string
    :param model_name: The name of the embedding model to use.
    :type model_name: string
    :param chunk_size: The size of text chunks for embedding comparison.
    :type chunk_size: int
    :return: The most relevant content chunk.
    :rtype: string
    """
    embedding_model = OllamaEmbeddingModel(model_name=model_name).get_model()

    # Generate embeddings for the query and text chunks
    query_embedding = embedding_model.embed_query(query)
    chunk_embeddings = embedding_model.embed_documents(chunked_text)

    # Compute cosine similarity between query embedding and chunk embeddings
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Find the chunk with the highest similarity
    most_relevant_index = np.argmax(similarities)
    most_relevant_chunk = chunked_text[most_relevant_index]

    return most_relevant_chunk


# # Example usage
# if __name__ == "__main__":
#     query = "What is the role of AI in healthcare?"
#     text = (
#         "Artificial Intelligence (AI) is transforming many industries, including healthcare. "
#         "In healthcare, AI is used for diagnostics, personalized medicine, and operational efficiency. "
#         "It enables faster and more accurate disease detection, helping doctors make better decisions."
#     )

#     relevant_content = find_relevant_content(query, text)
#     print("Most Relevant Content:", relevant_content)
