'''
Example Usage:
from utils.llm_model import LLMModel
llm_model = LLMModel().get_llm()
'''

from langchain_ollama import OllamaLLM
from llama_index.llms.langchain import LangChainLLM
# Use in RAG
class LLMModel:
    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str='llama3.1:8b-instruct-fp16', top_p:float=0.9, temperature:float=0.1, top_k:int=3):
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
                # max_tokens=max_tokens,
                num_ctx=15000
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
example_instructions = """Answer the question dimimited by triple backticks like a good kitten. ```{query}``` """ 

example_query = """Hi llama, would you like some lasagna?"""

response = OllamaLLMModel().query_ollama(example_query,example_instructions)
'''


# I cant seem to get queries with the other class without RAG
class OllamaLLMModel:

    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str='llama3.1:8b-instruct-fp16', top_p:float=0.9, temperature:float=0.1, top_k:int=3):
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
            # max_tokens=15000,
            num_ctx=15000
            )

    def query_ollama(self, query, instructions):
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(instructions) # Note: instructions str needs to include '{query}' for including query

        chain = prompt | self.llm_model

        return chain.invoke({"query": query})
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the provided text using the LLM's tokenizer.
        :param text: The text to tokenize.
        :return: The number of tokens in the text.
        """
        return self.llm_model.get_num_tokens(text)

