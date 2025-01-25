'''
Example usage:

from utils.doc_chunker import chunk_document
chunked_docs = chunk_document(documents)
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_document(documents, chunk_size=1000,chunk_overlap=200) -> list:
    """
    Splits a list of documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs
