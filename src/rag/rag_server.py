from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import chunk_document
from utils.query_rag import LocalDocumentIndexer

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "data/papers"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PDF_PATH = "data/papers/AlexNet.pdf"
LLM_MODEL_NAME = "llama3.2:1b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embed_model = EmbeddingModel(model_name=EMBED_MODEL_NAME).get_model()
llm_model = LLMModel(model_name=LLM_MODEL_NAME).get_llm()

# Pre-load default PDF and create query engine
documents = load_pdf(PDF_PATH)
chunked_docs = chunk_document(documents)
query_engine = LocalDocumentIndexer(
    embed_model=embed_model,
    llm_model=llm_model,
    documents=chunked_docs
).get_rag_query_engine()



@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handles PDF uploads and saves them to the data/papers directory."""
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    pdf_filename = secure_filename(pdf_file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    pdf_file.save(save_path)

    return jsonify({'message': 'File uploaded successfully', 'path': save_path})

@app.route('/api/query', methods=['POST'])
def query_pdf():
    """Handles querying a specific PDF."""

    global PDF_PATH

    data = request.get_json()
    pdf_path = data.get('pdf_path')
    query_text = data.get('query', '')

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'Invalid or missing pdf_path'}), 400

    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    if pdf_path != PDF_PATH:
        PDF_PATH = pdf_path
        documents = load_pdf(PDF_PATH)
        chunked_docs = chunk_document(documents)
        global query_engine
        query_engine = LocalDocumentIndexer(
            print("hi")
            embed_model=embed_model,
            llm_model=llm_model,
            documents=chunked_docs
        ).get_rag_query_engine()

    # Query the engine
    response = query_engine.query(query_text)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
