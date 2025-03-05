from utils.llm_service import init_engine , query_llm
import faiss
import ollama

"""
Leverage the embeddings from a given paper to average out the embeddings per section
to return a 'summary' of the paper.
"""

# pass faiss vector store for paper
# access embeddings
# filter according to the sections of metadata