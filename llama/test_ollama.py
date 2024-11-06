from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage

client = LlamaStackClient(
    base_url="http://localhost:5000",
)

response = client.inference.chat_completion(
    messages=[
        UserMessage(
            content="is TaxOnNeuro a good name?",
            role="user",
        ),
    ],
    model="Llama3.1-8B-Instruct",  # Specify the model identifier
    stream=False,
)
print(response)
