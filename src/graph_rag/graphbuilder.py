from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.globals import set_debug
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader


#set_debug(True)
import glob


def extract_knowledge_graph(file):
    # Load the document
    url="bolt://localhost:7687"
    username = "neo4j"
    password = "testtest"
    dbgraph = Neo4jGraph(url=url,username=username,password=password)
    embedding_provider = OllamaEmbeddings(model='llama3.1:latest')    
    loader = LLMSherpaFileLoader(
        file_path=file,
        new_indent_parser=True,
        apply_ocr=False,
        strategy="sections",
        llmsherpa_api_url="http://localhost:5001/api/parseDocument?renderFormat=all",
    )
    docs = loader.load_and_split()
    #for doc in docs:
        #print(doc)
        #input()
    new_vector = Neo4jVector.from_documents(
        docs,
        embedding_provider,
        graph=dbgraph,
        index_name="research",
        node_label="article",
        text_node_property="name",
        embedding_node_property="embedding",
        create_id_index=True,
    )

    # Extract the knowledge graph
    llm = ChatOllama(model='llama3.1:latest',tmeperature=0)#temperature=1,top_k=50,top_p=.5,repeat_penalty=1.2,num_ctx=128000)
    transformer = LLMGraphTransformer(llm=llm,strict_mode=True, ignore_tool_usage=True)
    

    graph = transformer.convert_to_graph_documents(docs)
    
    for document in graph:
        print(document.nodes)
        print(document.relationships)
        #print(document)
        #input()
    dbgraph.add_graph_documents(graph,baseEntityLabel=True,include_source=True)
    return graph

def load_and_extract_pdf_files(path="./papers2"):
    pdf_files = glob.glob(path+"/*.pdf")
    for file in pdf_files:
        extract_knowledge_graph(file)
    return pdf_files






if __name__ == "__main__":
    load_and_extract_pdf_files()
