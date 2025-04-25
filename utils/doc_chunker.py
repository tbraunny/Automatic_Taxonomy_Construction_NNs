# Example usage:
# from utils.doc_chunker import semantically_chunk_documents
# chunked_docs = semantically_chunk_documents(documents)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantically_chunk_documents(documents, embedder) -> list:
    """
    Semantically splits a list of documents into smaller chunks.

    :param documents: A list of documents to be split.
    :param embedder: A LangChain-compatible embedding model (e.g., OpenAIEmbeddings or OllamaEmbeddings)
    :return: A list of chunked documents.
    """
    text_splitter = SemanticChunker(embedder)
    return text_splitter.split_documents(documents)


def recursively_chunk_documents(documents, chunk_size=1000, chunk_overlap=200) -> list:
    """
    Splits a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    :param documents: A list of documents to be split.
    :param chunk_size: The size of each chunk. Defaults to 1000.
    :param chunk_overlap: The overlap between consecutive chunks. Defaults to 200.
    :return: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
