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
example_instructions = """Answer the question delimited by triple backticks like a good kitten. ```{query}``` """ 

example_query = """Hi llama, would you like some lasagna?"""

response = OllamaLLMModel().query_ollama(example_query,example_instructions)
'''

'''
This class assumes ollama is serving locally
'''


from langchain_core.prompts import ChatPromptTemplate
# I cant seem to do ollama queries with the using the class without RAG
# This query class is the solution
class OllamaLLMModel:

    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str='llama3.1:8b-instruct-fp16', top_p:float=0.9, temperature:float=0.1, top_k:int=3,num_ctx:int=10000,max_output_tokens:int=10000):
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
        self.num_ctx=num_ctx
        self.llm_model = OllamaLLM(
            model=model_name,
            top_p=top_p,
            top_k=top_k,
            # num_ctx=self.num_ctx,
            # max_tokens=max_output_tokens,
            temperature=temperature
            )

    def query_ollama(self, query:str, instructions:str, check_tokens:bool=False) -> str:
        prompt = ChatPromptTemplate.from_template(instructions) # Note: instructions str needs to include '{query}' for including query

        chain = prompt | self.llm_model

        if check_tokens:
            num_tokens = self.count_tokens(query + instructions)
            print(f"Query tokens: {num_tokens}\n")
            if num_tokens > self.num_ctx:
                print(f"\nTokens({num_tokens}) exceding Ctx window ({self.num_ctx})\n")

        return chain.invoke({"query": query})
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the provided text using the LLM's tokenizer.
        :param text: The text to tokenize.
        :return: The number of tokens in the text.
        """
        return self.llm_model.get_num_tokens(text)

