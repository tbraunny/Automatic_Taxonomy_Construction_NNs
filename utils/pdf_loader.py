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
        documents = []
        print("Loading PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF.")
        return documents
    else:
        print('\nPlease provide file path.')