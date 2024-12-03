from langchain_community.document_loaders import PyPDFLoader
import os
"""

from utils.pdf_loader import load_pdf
doc = load_pdf(file_path)


"""

def load_pdf(file_path: str, to_file: bool = False) -> list:
    """
    Loads the PDF into document objects and cleans the extracted text.
    Optionally writes the cleaned text to a file.
    :param file_path: Path to the PDF file.
    :param to_file: Whether to write the cleaned content to a file.
    :return: List of cleaned Document objects.
    :rtype: list
    """
    if not file_path:
        raise ValueError("File path must be provided.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Clean the page content of each document
    for doc in documents:
        doc.page_content = clean_extracted_text(doc.page_content)

    if to_file:
        # Combine the cleaned content into a single string
        cleaned_content = "\n\n".join(doc.page_content for doc in documents)
        write_to_text_file(file_path, cleaned_content)

    return documents

""" Helper functions """


def clean_extracted_text(text: str) -> str:
    """
    Cleans the extracted text by removing hyphens and unwanted line breaks.
    :param text: Text to process.
    :return: Cleaned text.
    """
    text = remove_hyphens(text)
    # text = remove_unwanted_line_breaks(text)
    return text

def remove_hyphens(text: str) -> str:
    """
    Removes hyphens that split words at line breaks and joins them correctly.
    """
    import re
    # Replace hyphens at the end of lines with the next line's start
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)
    # Remove standalone line breaks within paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text

def remove_unwanted_line_breaks(text: str) -> str:
    """
    Replaces line breaks within paragraphs with spaces.
    """
    return text.replace("\n", " ")

def write_to_text_file(file_path: str, content: str):
    """
    Writes the cleaned text content to a file in data/preprocessed/ 
    with the same base name as the original file.
    :param file_path: Original PDF file path.
    :param content: Cleaned text content to write.
    """
    # Get the base name without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = "data/preprocessed"
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write content to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Cleaned text written to {output_path}")
