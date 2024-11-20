'''
Example usage:
from utils.query_llm import query_llm

query = "What are the key differences between AlexNet and ResNet?"
response = query_llm(index, llm_predictor, query)
'''

from llama_index.core import VectorStoreIndex
class LLM:
    def __init__(self, index: VectorStoreIndex, llm_predictor):
        self.index =index
        self.llm_predictor=llm_predictor


##NEEDS TO GET RELEVANT STUFF FIRST 
    def query(self, query: str) -> str:
        """
        Query the LLM with a given query using the provided vector store index.
        :param index: A VectorStoreIndex instance.
        :param llm_predictor: The LLM predictor used for querying.
        :param query: The query string.
        :return: The response from the LLM as a string.
        """
        query_engine = self.index.as_query_engine(llm=self.llm_predictor)
        response = query_engine.query(query)
        return response.text if hasattr(response, "text") else str(response)
