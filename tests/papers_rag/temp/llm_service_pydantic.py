#!/usr/bin/env python
"""
test_chat_layer_details.py

This script demonstrates how to use structured outputs in your Advanced RAG Pipeline to
retrieve both the type and size of each layer of a neural network. Each layer is represented
as an object with two keys: 'type' (a string) and 'size' (an integer).

Ensure that:
- You have updated your Ollama Python library (pip install -U ollama).
- Your Ollama server is running.
- The pipeline module (llm_query.py) is available and exports init_engine, query_llm,
  LLMQueryEngine, and GENERATION_MODEL.
"""

import time
import logging
from pydantic import BaseModel
from utils.llm_service import init_engine, query_llm, LLMQueryEngine, GENERATION_MODEL

# Define a Pydantic model for each layer.
class Layer(BaseModel):
    type: str
    size: int

# Define a Pydantic model for the overall neural network details.
class NeuralNetworkDetails(BaseModel):
    layers: list[Layer]

# New structured-response version of generate_response.
def generate_response_structured(self, query: str, context_chunks: list) -> list:
    """
    Build the final prompt and call Ollama with a structured output.
    The prompt instructs the model to return a JSON object with a key 'layers' whose
    value is a list of objects. Each object should contain:
      - 'type': a string representing the type of the layer.
      - 'size': an integer representing the number of neurons (or filters, etc.) in that layer.
    """
    # Assemble evidence from context chunks.
    evidence_blocks = "\n\n".join(
        f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
        for chunk in context_chunks
    )
    full_prompt = (
        "Goal:\n"
        "Answer the technical query by fusing the evidence from the provided context.\n\n"
        f"Query: {query}\n\n"
        "Instructions:\n"
        "Return a JSON object with a key 'layers'. The value should be a list of objects, "
        "where each object represents one layer in the neural network. Each object must have exactly two keys: "
        "'type' (a string, e.g., 'Convolution', 'ReLU', 'Pooling', 'FullyConnected', etc.) and "
        "'size' (an integer representing the number of neurons/filters in that layer).\n"
        "Ensure that no additional text or keys are returned.\n\n"
        "Warnings:\n"
        "- Do not include any extra explanations, greetings, or disclaimers.\n"
        "- Use only the provided context to determine the answer.\n\n"
        "Context Dump:\n"
        f"{evidence_blocks}\n\n"
    )
    self.logger.info("Final prompt provided to LLM (structured):\n%s", full_prompt)
    
    # Use the structured output functionality via ollama.chat.
    from ollama import chat
    response = chat(
        messages=[{'role': 'user', 'content': full_prompt}],
        model=self.generation_model,
        format=NeuralNetworkDetails.model_json_schema()  # Enforce our JSON schema.
    )
    
    # Validate and parse the response using Pydantic.
    parsed = NeuralNetworkDetails.model_validate_json(response.message.content)
    return parsed.layers

# Monkey-patch the generate_response method of LLMQueryEngine with our new version.
LLMQueryEngine.generate_response = generate_response_structured

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize your engine (adjust the JSON file path as needed).
    json_file_path = "data/alexnet/doc_alexnet.json"
    engine = init_engine(
        model_name=GENERATION_MODEL,
        doc_json_file_path=json_file_path
    )
    
    # Define your query.
    query = (
        "Name each layer of this neural network sequentially. "
        "Do not generalize internal layers; include modification and activation layers. "
        "Also, specify the size of each layer (number of neurons or filters)."
    )
    
    # Run the query.
    start_time = time.time()
    layers = query_llm(GENERATION_MODEL, query, token_budget=1024)
    elapsed = time.time() - start_time
    
    print("Structured Layer Details:")
    for i, layer in enumerate(layers, start=1):
        layer = layer.dict()
        print(f"Layer {i}: Type = {layer['type']}, Size = {layer['size']}")
    print(f"Elapsed Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    print(NeuralNetworkDetails)
    main()
