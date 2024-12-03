'''
Example usage:

from utils.doc_chunker import chunk_document
chunked_docs = chunk_document(documents)
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_document(documents, chunk_size=800,chunk_overlap=300) -> list:
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
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

def chunk_document_for_nlm_LayoutPDFReader(document, chunk_size=1000, chunk_overlap=200) -> list:
    """
    Splits a single `Document` object from nlmatics's LayoutPDFReader into smaller chunks.

    Args:
        document: The parsed document object from LayoutPDFReader.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap size between chunks.

    Returns:
        List of chunked documents with metadata.
    """
    # Ensure the document has sections
    if not hasattr(document, "sections"):
        raise ValueError("The provided document does not have 'sections' attribute.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []

    for section in document.sections():
        section_title = section.title if section.title else "Untitled Section"
        section_content = []

        # Collect text from the section's children
        for child in section.children:
            if hasattr(child, "to_text"):
                section_content.append(child.to_text())

        # Combine all child content for this section
        section_text = "\n".join(section_content)

        # Split the section text into chunks
        if section_text.strip():  # Avoid processing empty sections
            chunks = text_splitter.create_documents(
                texts=[section_text],
                metadatas=[{"section_title": section_title}]
            )
            chunked_docs.extend(chunks)

    return chunked_docs










