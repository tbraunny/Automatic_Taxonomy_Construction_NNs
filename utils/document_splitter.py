'''
Example usage:

from utils.document_splitter import chunk_document
splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split(documents)
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(documents, chunk_size=1000,chunk_overlap=200) -> list:
    """
    Splits a list of documents into smaller chunks.
    :param documents: List of documents to split.
    :type documents: list
    :param chunk_size: Maximum size of each chunk.
    :type chunk_size: int
    :param chunk_overlap: Overlap size between chunks.
    :type chunk_overlap: int
    :return: List of document chunks.
    :rtype: list
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs