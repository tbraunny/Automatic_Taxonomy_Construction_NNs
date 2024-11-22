from flask import Flask, request, jsonify

from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import chunk_document
from utils.query_rag import LocalDocumentIndexer

clear

app = Flask(__name__)



PDF_PATH = "data/papers/AlexNet.pdf"
LLM_MODEL_NAME ="llama3.1:8b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"



@app.route('/api/query', methods=['POST'])
def query():

    data = request.get_json()
    pdf_path = data.get('pdf_path',PDF_PATH)
    llm_model_name = data.get('llm_model_name', LLM_MODEL_NAME)
    embed_model_name = data.get('embed_model_name', EMBED_MODEL_NAME)
    query_text = data.get('query', '')

    if pdf_path is  None:
        pdf_path=PDF_PATH
    if llm_model_name is None:
        llm_model_name=LLM_MODEL_NAME
    if embed_model_name is None:
        embed_model_name=EMBED_MODEL_NAME



    embed_model = EmbeddingModel(model_name=embed_model_name).get_model()
    llm_model = LLMModel(model_name=llm_model_name).get_llm()

    documents = load_pdf(pdf_path)
    chunked_docs = chunk_document(documents)
    
    query_engine = LocalDocumentIndexer(embed_model, llm_model,chunked_docs).get_rag_query_engine()

    response = query_engine.query(query_text)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
