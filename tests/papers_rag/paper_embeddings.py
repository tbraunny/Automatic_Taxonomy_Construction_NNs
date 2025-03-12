from utils.llm_service import init_engine , query_llm
from utils.llm_service import FAISSIndexManager
from utils.llm_service import LLMQueryEngine
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from utils.doc_chunker import semantically_chunk_documents
import json
import glob
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

"""
Leverage the embeddings from a given paper to discover possible
clustering techniques for taxonomy splits

Testing phase, check notes of each function for details on performance
"""

pdf_path = glob.glob("data/**/*.pdf" , recursive=True)
json_path = glob.glob("data/paper_embeddings/*.json" , recursive=True)
json_output = ("data/paper_embeddings")
log_file = "data/processed_pdfs.json"

def load_processed_pdfs():
    if os.path.exists(log_file):
        with open(log_file , "r") as f:
            return set(json.load(f))
    return set()

def save_processed_pdfs(processed_pdfs):
    with open(log_file , "w") as f:
        json.dump(list(processed_pdfs) , f)

def json_new_papers():
    processed_pdfs = load_processed_pdfs()
    new_pdfs = {pdf for pdf in pdf_path if pdf.endswith(".pdf") and pdf not in processed_pdfs}

    if new_pdfs:
        for pdf in new_pdfs:
            print("Entered")
            extract_filter_pdf_to_json(pdf, json_output)
            processed_pdfs.add(pdf)

        save_processed_pdfs(processed_pdfs)
    else:
        print("No new PDFs to process.")

def fetch_section_avg(docs: List):
    """
    Fetch the embeddings by section & average
    """

    json_path = glob.glob(f"{json_output}/*.json")
    vectore_store: List[np.ndarray] = []
    paper_names: List[str] = []

    for file in json_path:
        llm = LLMQueryEngine(file)
        vectore_store.append(llm.get_paper_representation())
        paper_names.append(os.path.basename(file).split('.')[0])  # Extract filename prefix
    vector_array = np.array(vectore_store)

    return vector_array , paper_names

def semantic_chunker(docs: List):
    chunked_docs = semantically_chunk_documents(docs)

def plot_data(vector_2d , labels , title , num_clusters=8):
    plt.figure(figsize=(10, 6), dpi=300)  # High DPI for clarity
    for i in range(num_clusters):
        cluster_points = vector_2d[labels == i]
        cluster_names = [paper_names[j] for j in range(len(labels)) if labels[j] == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.6, s=10)  # Small size, transparency
        
        for j, txt in enumerate(cluster_names):
            plt.annotate(txt, (cluster_points[j, 0], cluster_points[j, 1]), 
                        fontsize=5, alpha=0.6)  # Smaller font size

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)

    plt.savefig(f"data/plots/{title}.svg", format='svg', bbox_inches='tight') 
    print(f"Graphed via {title}")

def kmeans_pca(vector_array , paper_names):
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    vector_2d = pca.fit_transform(vector_array)

    # Apply K-Means clustering
    num_clusters = 8  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vector_array)
    labels = kmeans.labels_

    plot_data(vector_2d , labels , "pca_kmeans")

def kmeans_umap(vector_array , paper_names):
    reducer = umap.UMAP()
    vector_2d = reducer.fit_transform(vector_array)

    num_clusters = 8  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vector_array)
    labels = kmeans.labels_

    plot_data(vector_2d , labels , "umap_kmeans")

for file in json_path:
    print(file)
    docs = LLMQueryEngine.load_docs(file) # use @classmethod to fetch without instantiation
json_new_papers()
semantic_chunker(docs)
vector_array , paper_names = fetch_section_avg(docs) # refactor
#kmeans_pca(vector_array , paper_names)
#kmeans_umap(vector_array , paper_names)