"""
Example usage:

from src.pdf_extraction.extract_filter_pdf_to_pkl import extract_filter_pdf_to_pkl
extract_filter_pdf_to_pkl("data/resnet/resnet.pdf", "data/resnet/filtered_resnet.md")
"""

# Required dependencies: python-Levenshtein, fuzzywuzzy, langchain_core

import re
from fuzzywuzzy import fuzz
from langchain_core.documents.base import Document
from src.pdf_extraction.docling_pdf_loader import DoclingPDFLoader
from utils.pickle_utils import save_to_pickle

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
            
            # Normalize header by removing leading numbers and special characters
            header_text = re.sub(r'^[^a-zA-Z]+', '', header.strip("# \n")) # Match everything that is NOT a letter (numbers, periods, spaces, etc.) until the first letter appears
            
            if is_excluded_section(header_text, excluded_sections):
                filtered_content = filtered_content.replace(match.group(), '')

        # Clean up excess blank lines
        filtered_content = re.sub(r'\n{2,}', '\n', filtered_content).strip()
        
        # Store the cleaned content as a new Document object
        filtered_docs.append(Document(page_content=filtered_content, metadata=doc.metadata))
    
    return filtered_docs

# Writes file to txt for debuging before pickeling 
def write_list_to_txt(list, output_path):
    with open(output_path, "w") as f:
        for i, item in enumerate(list):
            print(f"Processing section {i + 1}")
            f.write(item.page_content + "\n---\n")

def extract_filter_pdf_to_pkl(pdf_path: str, output_path: str):
    """
    Extracts text from a PDF, filters out unwanted sections, and writes the result to a file.
    
    :param pdf_path: Path to the input PDF file.
    :param output_path: Path to save the extracted and filtered text.
    """
    loader = DoclingPDFLoader(file_path=pdf_path)
    docs = loader.load()
    
    filtered_docs = filter_sections_from_documents(docs, EXCLUDED_SECTIONS)

    # write_list_to_txt(filtered_docs, output_path)
    
    save_to_pickle(filtered_docs, output_path)
    
    print(f"Filtered PDF content processed and written to file: {output_path}")

# extract_filter_pdf_to_pkl("data/resnet/resnet.pdf", "data/resnet/filtered_resnet.pkl")
