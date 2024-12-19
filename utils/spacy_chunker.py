import spacy

"""
Example use:
from utils.spacy_chunker import spacy_chunking

chunks = spacy_chunking(text)

"""

def spacy_chunking(text:str, max_chunk_size=300)->list:
    """
    Chunk text into semantically coherent pieces using spaCy, optimized for research papers.
    :param text: Input text to chunk.
    :type text: string
    :param max_chunk_size: Maximum size of each chunk (in characters).
    :type max_chunk_size: int
    :return: List of text chunks.
    :rtype: list
    """
    nlp = spacy.load("en_core_web_trf")  
    doc = nlp(text)

    chunks = []
    current_chunk = ""

    for sent in doc.sents:
        # Add the sentence to the current chunk
        if len(current_chunk) + len(sent.text) <= max_chunk_size:
            current_chunk += " " + sent.text
        else:
            # Save the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sent.text

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1}: {chunk}")

    return chunks
