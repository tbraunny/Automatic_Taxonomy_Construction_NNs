# starting new:
#   - use ast to parse code
#    - look for specific functions in CNN (ex. 'Module' , '__init__' , 'forward')

#import ast

#from extract_annetto import prompt_engr

def convert_py_to_txt(file):
    with open(file) as f:
        data = f.read()
        f.close()

    with open("model_code.txt" , mode="w") as f:
        f.write(data)
        f.close()


# class CNN_parser(ast.NodeVisitor):
#     def __init__(self):
#         self.class_name = None
#         self.layers = []
#         self.forward_pass = []

#     def visit_classDef(self , node):
#         if any(base.id == 'Module' for base in node.bases if isinstance(base , ast.Attribute)):
#             self.class_name = node.name

#             for body_item in node.body:
#                 if (isinstance(body_item , ast.FunctionDef) and body_item.name == '__init__'):
#                     self.visit_init(self , node)
#                 elif (isinstance(body_item , ast.FunctionDef) and body_item.name == 'forward'):
#                     self.visit_forward_pass(self , node)
            
#     def visit_init(self , node):
#         for info in node.body:
#             if isinstance(info , ast.Assign) and isinstance(info.targets[0] , ast.Attribute):
#                 layer_name = info.target[0].attr # grab layer name
#                 layer_type = 0

#                 if isinstance(info.value , ast.Call): # if value of node is 
#                     print(info.value)
#                     if isinstance(info.value.func , ast.Attribute):
#                         layer_type = info.value.func.attr
#                     else:
#                         layer_type = info.value.func.id
#                     layer_values = [ast.dump(arg) for arg in info.value.args]

#                     self.layers.append((layer_name , layer_type , layer_values))

#     def visit_forward_pass(self , node):
#         for info in node.body:
#             if (isinstance(info , ast.Expr) and isinstance(info , ast.Call)):
#                 self.forward_pass.append(info.dump(info.value))

#     def parse_code(self , node):
#         if isinstance(node.bases == 'CNN'):
#             self.visit_classDef(self , node)
#         print("placeholder" , node)


# class CodeLoader:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.code = []

#     def load(self):
#         print("Loading Code...")
#         loader = PyPDFLoader(self.file_path)
#         self.documents = loader.load()
#         print(f"Loaded {len(self.documents)} pages from PDF.")
#         return self.documents

# class DocumentSplitter:
#     def __init__(self, chunk_size, chunk_overlap):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#     def split(self, documents):
#         print("Splitting documents into chunks...")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#         )
#         split_docs = text_splitter.split_documents(documents)
#         print(f"Split into {len(split_docs)} chunks.")
#         return split_docs

# class EmbeddingModel:
#     def __init__(self, model_name):
#         print("Initializing the embedding model...")
#         self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#         self.embed_model = LangchainEmbedding(self.embedding_model)
#         print("Embedding model initialized.")

#     def get_model(self):
#         return self.embed_model

# class LLMModel:
#     def __init__(self, model_name, top_p = 0.2, temperature=0.1, top_k=10):
#         print("Initializing the LLM...")
#         self.ollama_llm = OllamaLLM(
#             model=model_name,
#             top_p=top_p,
#             top_k=top_k,
#             temperature=temperature)
#         self.llm_predictor = LangChainLLM(llm=self.ollama_llm)
#         print("LLM initialized.")

#     def get_llm(self):
#         return self.llm_predictor

# class DocumentIndexer:
#     def __init__(self, embed_model, llm_predictor):
#         self.embed_model = embed_model
#         self.llm_predictor = llm_predictor
#         self.vector_index = None

#     def create_index(self, documents):
#         print("Creating LlamaIndex documents...")
#         index_documents = [Document(text=doc.page_content) for doc in documents]
#         print(f"Created {len(index_documents)} LlamaIndex documents.")

#         print("Building the VectorStoreIndex...")
#         self.vector_index = VectorStoreIndex.from_documents(
#             index_documents, 
#             embed_model=self.embed_model, 
#             llm_predictor=self.llm_predictor
#         )
#         print("VectorStoreIndex built.")
#         return self.vector_index

# def main():
#     # Load and process the PDF
#     pdf_loader = PDFLoader("rag/datasets/AlexNet.pdf")
#     documents = pdf_loader.load()

#     splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
#     split_docs = splitter.split(documents)

#     embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
#     llm_predictor = LLMModel(model_name="llama3.2:1b").get_llm()

#     indexer = DocumentIndexer(embed_model, llm_predictor)
#     vector_index = indexer.create_index(split_docs)

#     prompt_engr(vector_index, llm_predictor)

if __name__ == "__main__":
    convert_py_to_txt("test.py")

    path = '/file/path'
    '''
    for file in path:
        open(file)
        read(file)
    '''