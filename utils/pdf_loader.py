from langchain_community.document_loaders import PyPDFLoader

'''
Example usage:
from utils.pdf_loader import load_pdf

pdf_loader = PDFLoader("data/papers/AlexNet.pdf")
documents = pdf_loader.load()
'''

"""
A utility class to load PDF documents using LangChain's PyPDFLoader.
"""
def load_pdf(file_path=None):
    """
    Loads the PDF into document objects.
    :return: List of documents loaded from the PDF.
    :rtype: list
    """
    if file_path is not None:
        print("Loading PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF.")

        # Extract text from each document and save to a file
        with open("./loadpdf.txt", 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc.page_content + "\n\n")  # Write text content with blank line separation
        print("PDF content saved to loadpdf.txt")
        return documents
    else:
        print('\nPlease provide file path.')
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        return None
=======
=======
>>>>>>> Stashed changes

# def load_pdf(file_path=None):
#     """
#     Loads the PDF into document objects.
#     :return: List of documents loaded from the PDF.
#     :rtype: list
#     """
#     if file_path is not None:
#         documents = []
#         print("Loading PDF...")
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#         print(f"Loaded {len(documents)} pages from PDF.")
#         return documents
#     else:
#         print('\nPlease provide file path.')

import pdfplumber


def extract_text_from_pdf(file_path, output_file):
    paragraphs = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs.extend(text.split('\n\n'))  # Split into paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    # Write to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(paragraphs))  # Separate paragraphs by blank lines
    return paragraphs


import re


def clean_paragraphs(paragraphs, output_file):
    cleaned_paragraphs = []
    for para in paragraphs:
        para = re.sub(r'\s+', ' ', para)  # Remove extra whitespaces
        para = re.sub(r'\[\d+\]', '', para)  # Remove citation brackets like [1]
        if len(para.split()) > 5:  # Filter out very short paragraphs
            cleaned_paragraphs.append(para)
    
    # Write cleaned paragraphs to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned_paragraphs))  # Separate paragraphs by blank lines
    return cleaned_paragraphs

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(texts, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.create_documents(texts)




pdf_path = "data/papers/AlexNet.pdf"
load_pdf(pdf_path)

import re

# Sample input text
text = """[PASTE YOUR TEXT HERE]"""

# Regex to match titles (e.g., "1 Introduction", "3 The Architecture")
title_pattern = r"(\n\d+ [A-Z][^\n]*)"

# Split the text into sections
sections = re.split(title_pattern, text)

# Organize into title-content pairs
structured_sections = []
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    content = sections[i + 1].strip()
    structured_sections.append((title, content))

# Print or process the sections
for title, content in structured_sections:
    print(f"Title: {title}\n")
    print(f"Content:\n{content}\n")
    print("="*80)




# paragraphs = extract_text_from_pdf(pdf_path,"./raw.txt")


# cleaned_paragraphs = clean_paragraphs(paragraphs, "./clean.txt")



<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
