"""
This script extracts text from a PDF, filters out specific sections, and saves the processed content as a json file.

## Purpose:
1. **PDF to Document Conversion**: The script loads a PDF and converts it into a structured `Document` object.
2. **Filtering Unwanted Sections**: It removes sections based on a predefined list (e.g., References, Citations, Related Works) using fuzzy matching.
3. **Saving as Json**: The filtered `Document` objects are stored in a `.json` file including metadata.

## Why Use Document Objects Instead of Plain Text?
- **Metadata Retention**: Unlike plain text, `Document` objects keep metadata (e.g., page numbers, authors, etc.), making it easier to track information.
- **Structured Processing**: They enable advanced processing (e.g., chunking, indexing, and NLP applications) in downstream tasks.

## Example Usage:
```python 
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
extract_filter_pdf_to_json("data/resnet/resnet.pdf", "data/resnet/filtered_resnet.json")
"""

# Required dependencies: python-Levenshtein, fuzzywuzzy, langchain_core

import re
from fuzzywuzzy import fuzz
from langchain_core.documents.base import Document
from src.pdf_extraction.docling_pdf_loader import DoclingPDFLoader
from utils.document_json_utils import save_documents_to_json

EXCLUDED_SECTIONS = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

def is_excluded_section(header_text: str, excluded_sections: list, threshold: int = 80) -> bool:
    """
    Determines if a section header is similar to an excluded section using fuzzy matching.

    :param header_text: The extracted section header.
    :param excluded_sections: List of section headers to exclude.
    :param threshold: Similarity threshold (0-100). Higher values are stricter.
    :return: True if the section should be removed, False otherwise.
    """
    for excluded in excluded_sections:
        similarity = fuzz.ratio(header_text.lower(), excluded.lower())
        if similarity >= threshold:
            print(f"Removing '{header_text}' (Matched: {excluded}, Similarity: {similarity}%)")
            return True
    return False

def filter_sections_from_documents(documents: list, excluded_sections: list) -> list:
    """
    Filters out sections from documents by checking if their headers match excluded sections.
    
    :param documents: List of Document objects containing extracted text.
    :param excluded_sections: List of headers to exclude.
    :return: Filtered list of Document objects.
    """
    filtered_docs = []

    for doc in documents:
        content = doc.page_content
        pattern = r"(#{1,3}\s+.+?\n)(.*?)(?=## |\Z)"  # Match headers and sections
        matches = re.finditer(pattern, content, re.DOTALL)

        filtered_content = content


        for match in matches:
            header, section = match.groups()

            section = header + section
            
            # Normalize header by removing leading numbers and special characters
            header_text = re.sub(r'^[^a-zA-Z]+', '', header.strip("# \n")) # Match everything that is NOT a letter (numbers, periods, spaces, etc.) until the first letter appears
            
            if is_excluded_section(header_text, excluded_sections):
                continue

            # Clean up excess blank lines
            filtered_content = re.sub(r'\n{2,}', '\n', filtered_content).strip()

            # Enrich metadata
            new_metadata = {
                **doc.metadata,  # Preserve existing metadata
                "section_header": header_text
            }
            
            # Store the cleaned content as a new Document object
            filtered_docs.append(Document(page_content=section, metadata=new_metadata))
    
    return filtered_docs

# Writes file to txt for debuging before pickeling 
def write_list_to_txt(list, output_path):
    with open(output_path, "w") as f:
        for i, item in enumerate(list):
            f.write(item.page_content + "\n---\n")

def print_document_metadata(document: Document):
    """
    Prints all metadata from a Document object in a readable format.

    :param document: A Document object from LangChain.
    """
    print("\n=== Document Metadata ===")
    for key, value in document.metadata.items():
        print(f"{key}: {value}")
    print("=========================\n")


def extract_filter_pdf_to_json(pdf_path: str, output_path: str):
    """
    Extracts text from a PDF, filters out unwanted sections, and writes the result to a file.
    
    :param pdf_path: Path to the input PDF file.
    :param output_path: Path to save the extracted and filtered text.
    """
    loader = DoclingPDFLoader(file_path=pdf_path)
    docs = loader.load()
    
    filtered_docs = filter_sections_from_documents(docs, EXCLUDED_SECTIONS)

    # Save document to json for later use
    save_documents_to_json(filtered_docs, output_path)


    write_list_to_txt(filtered_docs, "wow.txt")
    
    # Example that doc -> json -> doc doesn't lose information

    # docs_from_json = load_documents_from_json("data/alexnet/doc_alexnet.json")

    # write_list_to_txt(docs_from_json, "wowza.md")

    # for doc in docs_from_json:
    #     print_document_metadata(doc)
    
    print(f"Filtered PDF content processed and written to file: {output_path}")

model_name = 'aae'
extract_filter_pdf_to_json(f"data/{model_name}/{model_name}.pdf", f"data/{model_name}/doc_{model_name}.json")
