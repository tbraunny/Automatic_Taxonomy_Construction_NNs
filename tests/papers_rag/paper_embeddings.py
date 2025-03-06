from utils.llm_service import init_engine , query_llm
from utils.llm_service import FAISSIndexManager
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
import faiss
import ollama

"""
Leverage the embeddings from a given paper to average out the embeddings per section
to return a 'summary' of the paper.
"""

# pass faiss vector store for paper
# access embeddings
# filter according to the sections of metadata

ann_name = "alexnet"
pdf_path = f"data/{ann_name}/{ann_name}.pdf"
json_file = f"data/{ann_name}/{ann_name}_doc.json"

fs = FAISSIndexManager()

init_engine(ann_name , json_file)
query_llm(ann_name , query="What neural network is this paper about?")
fs.compute_avg()