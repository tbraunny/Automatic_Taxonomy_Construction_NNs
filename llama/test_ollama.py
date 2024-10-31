from llama_stack_client import LlamaStackClient
import torch

class OllamaLLM:
    def __init__(self, model_name, host="localhost", port=5000):
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.inference.chat_completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False  # Set to True if you want streaming responses
        )
        return response['message']['content']

# Ensure CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Using GPU for inference.")

# Example usage
model_name = "llama3.1:8b-instruct-fp16"  # Use your specific model name
ollama_llm = OllamaLLM(model_name=model_name)

# Test the model
prompt = "Why is the sky blue?"
response = ollama_llm.generate(prompt)
print(response)
