import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from utils.file_utils import read_file , write_json_file

def py_to_md(path , out_path):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    code_chunks = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    with open(out_path , "w") as f:
        for doc in documents:
            f.write(doc.page_content)
            f.write("\n--\n")
    print(f"Extracted code written to {out_path} as markdown.")

    # use milvus database

def md_to_json(path , out_path):
    md_content = read_file(path)
    # FITFO from here


if __name__ == '__main__':
    model_name = 'resent'
    path = f"/home/richw/tom/ATCNN/data/{model_name}/{model_name}.py"
    md_path = f"/data/{model_name}/code_parsed_{model_name}.md"
    json_path = f"data/{model_name}/code_parsed_{model_name}.json"

    os.makedirs(os.path.dirname(md_path) , exist_ok=True)
    os.makedirs(os.path.dirname(json_path) , exist_ok=True)

    py_to_md(path , md_path)
    md_to_json(md_path , json_path)