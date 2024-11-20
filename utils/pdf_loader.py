from langchain_community.document_loaders import PyPDFLoader

'''
Example usage:
from utils.pdf_loader import PDFLoader

pdf_loader = PDFLoader("data/papers/AlexNet.pdf")
documents = pdf_loader.load()
'''

class PDFLoader:
    """
    A utility class to load PDF documents using LangChain's PyPDFLoader.
    """
    def __init__(self, file_path):
        """
        Constructor for loading PDF using LangChain.
        :param file_path: Path to the PDF file to load.
        :type file_path: string
        """
        self.file_path = file_path
        self.documents = []

    def load(self):
        """
        Loads the PDF into document objects.
        :return: List of documents loaded from the PDF.
        :rtype: list
        """
        print("Loading PDF...")
        loader = PyPDFLoader(self.file_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} pages from PDF.")
        return self.documents