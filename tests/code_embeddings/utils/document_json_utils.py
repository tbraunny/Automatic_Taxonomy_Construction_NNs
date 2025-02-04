import json
from langchain_core.documents.base import Document

def save_documents_to_json(documents: list, output_path: str):
    """
    Saves a list of Document objects to a JSON file.
    
    :param documents: List of Document objects.
    :param output_path: Path to save the JSON file.
    """
    json_data = []

    for doc in documents:
        json_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata  # Preserve metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Documents saved successfully to {output_path}")
    
def load_documents_from_json(input_path: str) -> list:
    """
    Loads a list of Document objects from a JSON file.
    
    :param input_path: Path to the JSON file.
    :return: List of reconstructed Document objects.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(input_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

            print(type(json_data))  # Should be a list
            print(json_data[:5])    # Print first 5 items (if it's a list)

    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in json_data
    ]

    print(f"Documents loaded successfully from {input_path}")
    return documents