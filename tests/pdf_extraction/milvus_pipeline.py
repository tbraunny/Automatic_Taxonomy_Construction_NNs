import os
from tempfile import TemporaryDirectory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Initialize embeddings
HF_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)

# Temporary directory for Milvus storage
MILVUS_URI = os.environ.get(
    "MILVUS_URI", f"{(tmp_dir := TemporaryDirectory()).name}/milvus_demo.db"
)

# Specify index parameters
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024},  # Adjust as needed
}

def create_vector_store(docs):
    """
    Creates a Milvus vector store from documents.
    :param docs: Iterable of split documents.
    :return: Milvus vector store instance.
    """
    return Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,
        index_params=index_params
    )
