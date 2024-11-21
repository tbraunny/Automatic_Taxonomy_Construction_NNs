from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import chunk_document
from utils.query_rag import LocalDocumentIndexer, RemoteDocumentIndexer

LLM_NAME = "llama3.2:1b"
EMBED_NAME = "all-MiniLM-L6-v2"

def local_query():    
    documents = load_pdf("data/papers/AlexNet.pdf")
    documents = chunk_document(documents=documents,chunk_size=1000, chunk_overlap=200)
    embed_model = EmbeddingModel(model_name=EMBED_NAME).get_model()
    llm_model = LLMModel(model_name=LLM_NAME).get_llm()
    query_engine = LocalDocumentIndexer(documents, embed_model=embed_model, llm_model=llm_model).get_rag_query_engine()
    response = query_engine.query("What is this paper about?")
    print(response)

def remote_query():
    pdf_path = './data/papers/AlexNet.pdf'
    ip_addr = '100.105.5.55'
    port = 5000
    query_engine = RemoteDocumentIndexer(pdf_path=pdf_path, device_ip=ip_addr, port=port).get_rag_query_engine()
    response = query_engine.query("What is this document about? What is alexnet")
    print(response)

# local_query()
remote_query()