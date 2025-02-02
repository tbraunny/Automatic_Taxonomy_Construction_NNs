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

def get_class_parent(cls: ThingClass) -> list:
    """
    Retrieves the direct parent classes of a given ontology class.
    
    :param cls: The ontology class (ThingClass) for which to find direct parents.
    :return: A list of direct parent classes, excluding any restrictions.
    """
    return [parent for parent in cls.is_a if isinstance(parent, ThingClass)]

    with open(input_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in json_data
    ]

    print(f"Documents loaded successfully from {input_path}")
    return documents
