'''
Example Usage:
from utils.llm_model import LLMModel
llm_model = LLMModel().get_llm()
'''

from langchain_ollama import OllamaLLM
from llama_index.llms.langchain import LangChainLLM
# For use only in RAG
class LLMModel:
    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str='llama3.1:8b', top_p:float=0.9, temperature:float=0.1, top_k:int=3):
        """
        Constructor for LLM model.
        :param model_name: Name of the LLM model.
        :type model_name: string
        :param top_p: Top-p (nucleus sampling) value for generation.
        :type top_p: float
        :param temperature: Sampling temperature for generation.
        :type temperature: float
        :param top_k: Top-k sampling parameter for generation.
        :type top_k: int
        """
        self.llm_model = LangChainLLM(
            llm=OllamaLLM(
                model=model_name,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_tokens=128000,
                num_ctx=128000
                )
            )

    def get_llm(self):
        """
        Retrieves the LLM predictor object.
        :return: The LLM predictor object.
        :rtype: LangChainLLM
        """
        return self.llm_model
    


    
'''
Example Usage:
from utils.llm_model import OllamaLLMModel

# Note: instructions str needs to include '{query}' for including query
example_instructions = """Answer the question delimited by triple backticks like a good kitten. ```{query}``` """ 

example_query = """Hi llama, would you like some lasagna?"""

response = OllamaLLMModel().query_ollama(example_query,example_instructions)
'''

'''
This class assumes ollama is serving locally
'''


# I cant seem to do ollama queries with the using the class without RAG
# This query class is the solution
class OllamaLLMModel:

    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str='llama3.1:8b', top_p:float=0.9, temperature:float=0.1, top_k:int=3):
        """
        Constructor for LLM model.
        :param model_name: Name of the LLM model.
        :type model_name: string
        :param top_p: Top-p (nucleus sampling) value for generation.
        :type top_p: float
        :param temperature: Sampling temperature for generation.
        :type temperature: float
        :param top_k: Top-k sampling parameter for generation.
        :type top_k: int
        """
        self.llm_model = OllamaLLM(
            model=model_name,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=128000,
            num_ctx=128000
            )

    def query_ollama(self, query:str, instructions:str) -> str:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(instructions) # Note: instructions str needs to include '{query}' for including query

        chain = prompt | self.llm_model

        return chain.invoke({"query": query})