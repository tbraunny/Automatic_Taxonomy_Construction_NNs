#Install Ollama in wsl: curl -fsSL https://ollama.com/install.sh | sh

#Hint: allocate max cpu cores and ram

#First, start ollama server: ollama server
#Ensure ollama is loaded: ls /home/richw/.ollama/models
#       if not, pull: ollama pull llama3
#


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding

from utils.query_llm import prompt_engr

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = []

    def load(self):
        print("Loading PDF...")
        loader = PyPDFLoader(self.file_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} pages from PDF.")
        return self.documents

class DocumentSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        return split_docs

class EmbeddingModel:
    def __init__(self, model_name):
        print("Initializing the embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.embed_model = LangchainEmbedding(self.embedding_model)
        print("Embedding model initialized.")

    def get_model(self):
        return self.embed_model

class LLMModel:
    def __init__(self, model_name, top_p = 0.2, temperature=0.1, top_k=10):
        print("Initializing the LLM...")
        self.ollama_llm = OllamaLLM(
            model=model_name,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature)
        self.llm_predictor = LangChainLLM(llm=self.ollama_llm)
        print("LLM initialized.")

    def get_llm(self):
        return self.llm_predictor

class DocumentIndexer:
    def __init__(self, embed_model, llm_predictor):
        self.embed_model = embed_model
        self.llm_predictor = llm_predictor
        self.vector_index = None

    def create_index(self, documents):
        print("Creating LlamaIndex documents...")
        index_documents = [Document(text=doc.page_content) for doc in documents]
        print(f"Created {len(index_documents)} LlamaIndex documents.")

        print("Building the VectorStoreIndex...")
        self.vector_index = VectorStoreIndex.from_documents(
            index_documents, 
            embed_model=self.embed_model, 
            llm_predictor=self.llm_predictor
        )
        print("VectorStoreIndex built.")
        return self.vector_index

def main():
    # Load and process the PDF
    pdf_loader = PDFLoader("rag/datasets/AlexNet.pdf")
    documents = pdf_loader.load()

    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split(documents)

    embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
    llm_predictor = LLMModel(model_name="llama3.2:1b").get_llm()

    indexer = DocumentIndexer(embed_model, llm_predictor)
    vector_index = indexer.create_index(split_docs)

    prompt_engr(vector_index, llm_predictor)

if __name__ == "__main__":
    main()