from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
Example usage:

from utils.document_splitter import DocumentSplitter
splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split(documents)
'''

class DocumentSplitter:
    """
    A utility class for splitting documents into smaller chunks.
    """
    def __init__(self, chunk_size, chunk_overlap):
        """
        Constructor for document splitter.
        :param chunk_size: Maximum size of each chunk.
        :type chunk_size: int
        :param chunk_overlap: Overlap size between chunks.
        :type chunk_overlap: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        """
        Splits a list of documents into smaller chunks.
        :param documents: List of documents to split.
        :type documents: list
        :return: List of document chunks.
        :rtype: list
        """
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        return split_docs
