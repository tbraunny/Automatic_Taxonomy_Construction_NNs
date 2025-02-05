# Example usage:
# from utils.doc_chunker import semantically_chunk_documents
# chunked_docs = semantically_chunk_documents(documents)


def semantically_chunk_documents(documents, ollama_model="bge-m3:latest") -> list:
    from langchain_experimental.text_splitter import SemanticChunker
    import ollama

    class OllamaEmbeddings:
        def embed_documents(self, texts):
            # texts is a list of strings
            return [
                ollama.embeddings(model=ollama_model, prompt=text)["embedding"]
                for text in texts
            ]
    """
    Semantically splits a list of documents into smaller chunks.
    
    :param documents: A list of documents to be split.
    :return: A list of chunked documents.
    """
    embedder = OllamaEmbeddings()
    text_splitter = SemanticChunker(embedder)

    # Split the documents into smaller chunks
    chunked_docs = text_splitter.split_documents(documents)

    # Maybe be better to convert to one text, split, then piece back to doc, probably not though
    # texts = [doc.page_content for doc in documents]
    # full_text = " ".join(texts)
    # chunked_texts = text_splitter.split_text(full_text)
    # chunked_docs = [Document(page_content=chunk) for chunk in split_texts]

    return chunked_docs

def recursively_chunk_documents(documents, chunk_size=1000, chunk_overlap=200) -> list:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    """
    Splits a list of documents into smaller chunks.
    
    :param documents: A list of documents to be split.
    :param chunk_size: The size of each chunk. Defaults to 1000.
    :param chunk_overlap: The overlap between consecutive chunks. Defaults to 200.
    :return: A list of chunked documents.
    """
    # Initialize the text splitter with the specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the documents into smaller chunks
    chunked_docs = text_splitter.split_documents(documents)
    
    return chunked_docs
