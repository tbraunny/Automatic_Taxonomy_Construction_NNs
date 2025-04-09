"""
Let's test out this mf'in bert

NOTES: I don't think this will work, the semantic space differs from the langchain model (384 dim vs. 768 dim, or something like that).
       in order to use both code & pdf in same chroma db either use CodeX for both extractions, or convert code to JSON
"""

"""
DEPRECATED
"""

import chromadb
from transformers import RobertaTokenizer, RobertaModel
import torch
from sentence_transformers import SentenceTransformer
#import PyPDF2
from sklearn.decomposition import PCA
import numpy as np

# Initialize ChromaDB client
client = chromadb.Client()

# Initialize models for embedding
code_model = RobertaModel.from_pretrained("microsoft/codebert-base")
code_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
pdf_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Sentence-BERT for PDF text

# Function to embed Python code
def embed_code(code):
    inputs = code_tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = code_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()

# Function to embed PDF content
def embed_pdf(pdf_text):
    return pdf_model.encode(pdf_text).flatten()

# Add embeddings to ChromaDB
def store_embeddings_in_chroma(embeddings, metadata, collection_name="embeddings_collection"):
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        documents=[metadata['text']],
        metadatas=[metadata],
        embeddings=[embeddings],
        ids=[metadata['id']]
    )

# Example: Embed and store Python code
code_file = '/home/richw/tom/ATCNN/data/alexnet/alexnet.py'
with open(code_file, 'r') as f:
    code_content = f.read()

code_embeddings = embed_code(code_content)
code_metadata = {"id": "code_1", "source": "python_code", "text": code_content}
store_embeddings_in_chroma(code_embeddings, code_metadata)

# Example: Embed and store PDF content
# pdf_file = 'example_file.pdf'
# with open(pdf_file, 'rb') as f:
#     reader = PyPDF2.PdfReader(f)
#     pdf_text = ' '.join([page.extract_text() for page in reader.pages])

# pdf_embeddings = embed_pdf(pdf_text)
# pdf_metadata = {"id": "pdf_1", "source": "pdf_file", "text": pdf_text}
# store_embeddings_in_chroma(pdf_embeddings, pdf_metadata)

# Example: Query ChromaDB and retrieve similar code and PDF embeddings
def retrieve_similar_embeddings(query, collection_name="embeddings_collection"):
   

    collection = client.get_collection(collection_name)
    query_embedding = embed_pdf(query)  # Use the appropriate embedding function for your query (code or pdf)
    query_embedding_resized = pad_embedding(query_embedding, 768)
    results = collection.query(
        query_embeddings=[query_embedding_resized],
        n_results=1
    )
    return results

from sklearn.decomposition import PCA

def resize_embedding(embedding, target_dim):
    pca = PCA(n_components=target_dim)
    return pca.fit_transform(np.array(embedding).reshape(1, -1))

import numpy as np

def pad_embedding(embedding, target_dim):
    current_dim = len(embedding)
    if current_dim < target_dim:
        # Pad with zeros or any other strategy for padding
        padding = np.zeros(target_dim - current_dim)
        return np.concatenate([embedding, padding])
    return embedding


# Example Query
query = "How does the forward pass of the given architecture in python code prevent overfitting?"
results = retrieve_similar_embeddings(query)
print(results)