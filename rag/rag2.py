from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaLLM
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from llama_index.core import Document
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding

from ontology_handler import OntologyHandler
import os

print("AAAAA")

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
        print("Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.embed_model = LangchainEmbedding(self.embedding_model)
        print("Embedding model initialized.")

    def get_model(self):
        return self.embed_model

class LLMModel:
    def __init__(self, model_name, top_p=0.2, temperature=0.1, top_k=10):
        print("Initializing LLM...")
        self.ollama_llm = OllamaLLM(
            model=model_name,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature)
        self.llm_predictor = LangChainLLM(llm=self.ollama_llm)
        print("LLM initialized.")

    def get_llm(self):
        return self.llm_predictor

def prompt_engr(llm_predictor, pdf_context, ontology_context):
    print("Generating response using the RAG system...")
    query = f"""
    Based on the following PDF context:
    {pdf_context}

    And the following ontology context:
    {ontology_context}

    Generate a response to the user's query about the relationships and structure.
    """
    response = llm_predictor.predict(query)
    print("Response:")
    print(response)

def main():
    # Load and process the PDF
    pdf_loader = PDFLoader("rag/datasets/AlexNet.pdf")
    documents = pdf_loader.load()

    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split(documents)

    # Prepare embedding and LLM models
    embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
    llm_predictor = LLMModel(model_name="llama3").get_llm()

    # Prepare ontology handler
    ontology_file = "rag/ontologies/alexnet.owl"
    graphdb_endpoint = "http://localhost:7200"
    repository_name = "ontology_repo"

    ontology_handler = OntologyHandler(endpoint_url=graphdb_endpoint, repository_name=repository_name)
    if not os.path.exists(".ontology_loaded"):
        ontology_handler.load_ontology(ontology_file)
        with open(".ontology_loaded", "w") as marker:
            marker.write("Ontology loaded.")

    # Retrieve context from ontology
    ontology_query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object.
    }
    LIMIT 10
    """
    ontology_context = ontology_handler.query_ontology(ontology_query)

    # Combine PDF and ontology context into the prompt
    pdf_context = "\n".join([doc.page_content for doc in split_docs])
    prompt_engr(llm_predictor, pdf_context, ontology_context)

if __name__ == "__main__":
    main()
