from langchain_ollama import OllamaLLM
from llama_index.llms.langchain import LangChainLLM

'''
Example Usage:
from utils.llm_model import LLMModel
llm_predictor = LLMModel(model_name="llama3.2:1b").get_llm()
'''
MODEL_NAME = 'llama3.2:1b'

class LLMModel:
    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name=MODEL_NAME, top_p=0.9, temperature=0.1, top_k=3, max_tokens=10):
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
        print("Initializing the LLM...")
        self.ollama_llm = OllamaLLM(
            model=model_name,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens)
        self.llm_predictor = LangChainLLM(llm=self.ollama_llm)
        print("LLM initialized.")

    def get_llm(self):
        """
        Retrieves the LLM predictor object.
        :return: The LLM predictor object.
        :rtype: LangChainLLM
        """
        return self.llm_predictor
