from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import DocumentSplitter
from utils.query_rag import DocumentIndexer

LLM_NAME = "llama3.2:1b"
EMBED_NAME = "all-MiniLM-L6-v2"

def main():    

    #Loads PDF to 
    documents = load_pdf("data/papers/AlexNet.pdf")

    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split(documents)

    embed_model = EmbeddingModel(model_name=EMBED_NAME).get_model()
    llm_predictor = LLMModel(model_name=LLM_NAME).get_llm()

    indexer = DocumentIndexer(embed_model, llm_predictor)
    vector_index = indexer.create_index(split_docs)

if __name__ == "__main__":
    main()