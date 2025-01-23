from langchain_docling import DoclingLoader, ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from pathlib import Path
from tempfile import mkdtemp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define constants
FILE_PATH = "/path/to/your/document.pdf"  # Replace with your document's path
EXPORT_TYPE = ExportType.DOC_CHUNKS  # Or ExportType.MARKDOWN if Markdown export is needed
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_API_KEY")
MILVUS_URI = str(Path(mkdtemp()) / "docling_demo.db")
QUESTION = "What methods does this architecture use to combat overfitting?"

# Initialize DoclingLoader
loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=EXPORT_TYPE
)

# Load documents
docs = loader.load()

# Determine splits
splits = docs if EXPORT_TYPE == ExportType.DOC_CHUNKS else [doc.page_content for doc in docs]

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

# Set up vector store
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embedding,
    collection_name="docling_demo",
    connection_args={"uri": MILVUS_URI},
    index_params={"index_type": "FLAT"},
    drop_old=True,
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN,
)

# Define prompt template
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"
)

# Create RAG pipeline
question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Perform query
response = rag_chain.invoke({"input": QUESTION})

# Output results
print("Question:", QUESTION)
print("Answer:", response["answer"])

# Save extracted content
output_path = "/path/to/output/extracted_content.md"
with open(output_path, "w") as f:
    for doc in docs:
        f.write(doc.page_content)
        f.write("\n---\n")
print(f"Extracted content saved to {output_path}")
