from utils.llm_service import init_engine , query_llm
from utils.llm_service import FAISSIndexManager
from utils.llm_service import LLMQueryEngine
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from utils.document_json_utils import load_documents_from_json
import faiss
import ollama
import glob
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""
Leverage the embeddings from a given paper to average out the embeddings per section
to return a 'summary' of the paper.
"""

pdf_path = glob.glob("data/**/*.pdf" , recursive=True)
json_path = glob.glob("data/**/*.json" , recursive=True)
json_output = ("data/paper_embeddings")

#print(pdf_path)

# for file in pdf_path:
#     print("PDF: " , file)
#     extract_filter_pdf_to_json(file , json_output)

json_path = glob.glob(f"{json_output}/*.json")
vectore_store: List[np.ndarray] = []
paper_names: List[str] = []

for file in json_path:
    llm = LLMQueryEngine(file)
    vectore_store.append(llm.get_paper_representation())
    paper_names.append(os.path.basename(file).split('.')[0])  # Extract filename prefix

#print(vectore_store)

vector_array = np.array(vectore_store)  # Shape: (num_samples, 1024)

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
vector_2d = pca.fit_transform(vector_array)  # Shape: (num_samples, 2)

# Apply K-Means clustering
num_clusters = 8  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vector_array)
labels = kmeans.labels_

# # Visualization
# plt.figure(figsize=(10, 6))
# for i in range(num_clusters):
#     cluster_points = vector_2d[labels == i]
#     cluster_names = [paper_names[j] for j in range(len(labels)) if labels[j] == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
#     for j, txt in enumerate(cluster_names):
#         plt.annotate(txt, (cluster_points[j, 0], cluster_points[j, 1]), fontsize=8, alpha=0.7)

# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.title("Paper Representations Clustered by KMeans")
# plt.show() # USELESS
# plt.savefig("data/plot.png")

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
plt.title("Paper Representations Clustered by KMeans")
#plt.legend(markerscale=2, fontsize=6)  # Smaller legend text

plt.savefig("data/plot.svg", format='svg', bbox_inches='tight') 

print("GRaph?")



# ann_name = "alexnet"
# pdf_path = f"data/{ann_name}/{ann_name}.pdf"
# json_file = f"data/{ann_name}/{ann_name}_doc.json"

# fs = FAISSIndexManager()
# llm = LLMQueryEngine(json_file)

# #init_engine(ann_name , json_file)
# paper_representation = llm.get_paper_representation(ann_name)