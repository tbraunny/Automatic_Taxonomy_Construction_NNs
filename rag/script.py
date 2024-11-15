from utils.pdf_loader import PDFLoader
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import DocumentSplitter
from utils.document_indexer import DocumentIndexer
from utils.placeholder_prompts import PromptEngr

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

    extractor = PromptEngr(vector_index,llm_predictor)
    extractor.save_info_to_file("placeholder_network_info.json")

if __name__ == "__main__":
    main()