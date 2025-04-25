"""
This script extracts text from a PDF, filters out unwanted sections based on fuzzy matching,
and saves the processed content as a JSON file. The title and each section’s header is stored in the document metadata.
Document objects are stored as json for persistence.

Example usage:
    python3 extract_filter_pdf_to_json.py --pdf_path input_pdf_name.pdf
        i.e. python3 src/pdf_extraction/extract_filter_pdf_to_json.py --pdf_path data/alexnet/alexnet.pdf

    Note: After extraction, load_documents_from_json() from utils/document_json_utils.py can be used to load the JSON file back into Document objects.
"""

import re
import argparse
import logging
logger = logging.getLogger(__name__)
logging.getLogger("docling").setLevel(logging.ERROR)

import json
from fuzzywuzzy import fuzz
from langchain_core.documents.base import Document
from src.pdf_extraction.utils.docling_pdf_loader import DoclingPDFLoader
from langchain_core.documents.base import Document
# from utils.logger_util import get_logger
# logger = get_logger("pdf_extraction")

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

    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in json_data
    ]

    print(f"Documents loaded successfully from {input_path}")
    return documents

### Utils for debugging
def write_list_to_txt(list, output_path = 'pdf_extract_test.txt'):
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

# List of section names to exclude (using fuzzy matching)
EXCLUDED_SECTIONS = [
    "References", "Citations", "Related Works", "Authors", "Background"
]

def is_excluded_section(header_text: str, excluded_sections: list, threshold: int = 80) -> bool:
    """
    Determines if a section header is similar to one of the excluded section names using fuzzy matching.
    
    :param header_text: The extracted header text.
    :param excluded_sections: List of section headers to exclude.
    :param threshold: Similarity threshold (0-100); higher values are stricter.
    :return: True if the header should be excluded, False otherwise.
    """
    for excluded in excluded_sections:
        similarity = fuzz.ratio(header_text.lower(), excluded.lower())
        if similarity >= threshold:
            logger.info(f"Excluding section '{header_text}' (Matched: {excluded}, Similarity: {similarity}%)")
            return True
    return False

def filter_sections_from_documents(documents: list, excluded_sections: list) -> list:
    """
    Splits the content of each Document into sections based on markdown-style headers(i.e. "## 1. Introduction").
    The header (i.e. the first line starting with '#' characters) is removed from the main text and
    stored in the document metadata as 'section_header'. Sections whose header matches one of the 
    excluded sections (using fuzzy matching) are dropped.
    
    :param documents: List of Document objects (each with a 'page_content' string).
    :param excluded_sections: List of section header names to filter out.
    :return: List of filtered Document objects.
    """
    filtered_docs = []
    # Regex pattern to match a header (with 1-3 '#' symbols) and capture the following section text.
    pattern = r"(#{1,3}\s+.+?\n)(.*?)(?=(?:#{1,3}\s)|\Z)"
    
    for doc in documents:
        content = doc.page_content
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if not matches:
            # No sections found: add the document as-is.
            filtered_docs.append(doc)
            continue
        first_doc = True
        title = None
        for match in matches:
            header, body = match.groups()

            # Keep header with its body text
            body = header + body
            
            # Clean the header; strip leading '#' and non-letter characters.
            header_text = re.sub(r'^[^a-zA-Z]+', '', header.strip("# \n"))

            if first_doc:
                title = header_text
                first_doc = False
                continue
            
            if is_excluded_section(header_text, excluded_sections):
                continue

            # Create metadata
            new_metadata = {**doc.metadata, "section_header": header_text, "title":title, "type":"paper"}

            filtered_docs.append(Document(page_content=body.strip(), metadata=new_metadata))
    
    return filtered_docs

def write_list_to_txt(documents: list, output_path: str) -> None:
    """
    Writes the page content of each Document object to a text file for debugging.
    
    :param documents: List of Document objects.
    :param output_path: Path to save the debug text file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.page_content + "\n---\n")

def extract_filter_pdf_to_json(pdf_path: str, debug: bool = False) -> None:
    """
    Loads a PDF, extracts and filters its text into sections, and saves the results as a JSON file.
    
    :param pdf_path: Path to the input PDF file.
    :param output_path: Path to save the JSON output.
    """
    logger.info(f"Loading PDF from: {pdf_path}")
    loader = DoclingPDFLoader(file_path=pdf_path)
    docs = loader.load()

    if not pdf_path:
        raise ValueError("PDF path is required.")
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Input file must be a PDF.")
    
    output_path = pdf_path.replace(".pdf" , "_pdf.json")
    logger.info("Filtering sections from extracted documents...")
    filtered_docs = filter_sections_from_documents(docs, EXCLUDED_SECTIONS)
        
    logger.info(f"Saving filtered documents to JSON: {output_path}")
    save_documents_to_json(filtered_docs, output_path)

    logger.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract, filter, and convert PDF content to JSON."
    )
    parser.add_argument("--pdf_path", required=True ,type=str, help="Path to the input PDF file.")
    args = parser.parse_args()
    if not args.pdf_path:
        raise ValueError("PDF path is required.")
    
    extract_filter_pdf_to_json(args.pdf_path)
# python3 src/pdf_extraction/extract_filter_pdf_to_json.py --pdf_path data/transformer/transformer.pdf
