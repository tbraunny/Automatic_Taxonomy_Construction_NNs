from ollama import chat
from typing import List
from utils.document_json_utils import load_documents_from_json

# TODO: Implement pydantic for json output

def verify_user_file_inputs(list_json_doc_paths: List[str]) -> List[bool]:
    """Verifies if each document is about neural network architectures and training styles."""
    
    results = []
    
    for j in list_json_doc_paths:
        doc_content = load_documents_from_json(j)

        # Send document content to the model for classification
        response = chat(
            model='qwen2.5:32b',
            messages=[
                {'role': 'system', 'content': 'You are an expert in deep learning and neural networks.'},
                {'role': 'user', 'content': f'Analyze the following text and determine if it is about neural networks in any sense. '
                                            f'Respond only with "True" or "False".\n\nDocument:\n{doc_content}'}
            ],
            stream=False,
        )
        
        result_text = response['message']['content'].strip()
        results.append(result_text.lower() == "true")
    
    return results

if __name__ == "__main__":
    response = verify_user_file_inputs(["data/vgg16/doc_vgg16.json","data/placebo/placebo.json"])
    print (response)