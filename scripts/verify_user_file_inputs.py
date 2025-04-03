from typing import TypeVar, Generic, Literal, Optional, List,Union

from pydantic import BaseModel, Field, field_validator
from ollama import chat

from utils.document_json_utils import load_documents_from_json


# TODO: Implement pydantic for json output

# Define a generic response model for structured LLM output.
T = TypeVar("T")
class LLMResponse(BaseModel, Generic[T]):
    answer: T

class VerifyUserFileInputsDetails(BaseModel):
    is_neural_network: bool = Field(..., description="Whether the document is about neural networks.")

class VerifyUserFileInputsResponse(LLMResponse[VerifyUserFileInputsDetails]):
    pass

def verify_user_file_inputs(list_json_doc_paths: List[str]) -> List[bool]:
    """Verifies if each document is about neural network architectures and training styles."""
    
    results = []
    
    for j in list_json_doc_paths:
        doc_content = load_documents_from_json(j)

        query = "Is this document about neural networks?"

        verify_json_format_prompt = (
            f'Respond only with "True" or "False".\n'
            f""""Return the task name in JSON format with the key 'answer'.\n"""
            f"""If the document is about neural networks, the output should be "True" and structured as follows:\n\n"""
            "{\n"
            '    "answer": "True"\n'
            "}\n\n"
        )
        # Build the full prompt by combining the query, JSON format instructions, warnings, and the evidence blocks.
        full_prompt = (
            "Goal:\n"
            "Answer the technical query by fusing the evidence from the provided context.\n\n"
            f"Query: {query}\n\n"
            "Instructions:\n"
            f"{verify_json_format_prompt}\n\n"
            "Ensure that no additional text or keys are returned.\n"
            "Warnings:\n"
            "- Do not include extra explanations, greetings, or disclaimers.\n"
            "- Use only the provided context to answer the question.\n\n"
            "The Text to analyze:\n"
            f"{doc_content}\n\n"
        )

        # Send document content to the model for classification
        response = chat(
            model='qwen2.5:32b',
            messages=[
                {'role': 'system', 'content': 'You are an expert in deep learning and neural networks. \n'},
                {'role': 'user', 'content': prompt},
            ],
            stream=False,
            format=VerifyUserFileInputsResponse.model_json_schema()
        )
        result_text = response['message']['content'].strip()
        results.append(result_text.lower() == "true")
    
    return results

if __name__ == "__main__":
    response = verify_user_file_inputs(["data/vgg16/doc_vgg16.json","data/placebo/placebo.json"])
    print (response)