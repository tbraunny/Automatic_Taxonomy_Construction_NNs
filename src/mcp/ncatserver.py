import rdflib
import subprocess
from mcp.server.fastmcp import FastMCP
import glob
import json
import owlready2
from owlready2 import *
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from rdflib import Graph, URIRef
import os
import time
import sys
from pathlib import Path
os.environ['KERAS_BACKEND'] = 'torch'

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.llm_service import init_engine, query_llm
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from pydantic import BaseModel
from src.main import main
from utils.constants import Constants as C

#from ... import *
#from taxonomy import llm_service,criteria,create_taxonomy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.taxonomy import llm_service,create_taxonomy
from src.graph_rag.graphRAG import neo4j_connection, get_answer_for_question 
from src.graph_rag.insertion import import_rdf, get_neo4j_connection 

# Create an MCP server named "OWL Server
mcp = FastMCP("NCAT Server")

paperpath = 'data/more_papers'
options = []

# pydantic chat response
class ChatResponse(BaseModel):
    answer: str

@mcp.tool()
def list_available_titles() -> str:
    """List the available papers and their titles. Returns a json dictionary with titles as key and ann name as value."""
    unique_titles = set()
    title_to_path = {}
    json_doc_paths = glob.glob(f"{paperpath}/**/*doc*.json" , recursive=True) # Grabs all pdf doc json's change to pdf
    for file in json_doc_paths:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if type(entry) == dict:
                    title = entry.get("metadata", {}).get("title")
                    if title:
                        unique_titles.add(title)
                        annname = os.path.dirname(file).split('/')[-1]
                        index = options.index(os.path.dirname(file))
                        title_to_path[title] = { 'annname': annname, "choice": index}
    return str(title_to_path)

@mcp.tool()
def list_available_directories():
    """
    Lists the available directories that can be used to create json files. Which ones have json associated them that this tool can do queries on.
    """
    global options
    options = []
    returned_options = {}
    for index,path in enumerate(Path(paperpath).rglob("*")):
        if path.is_dir():
            print(path)
            options.append(str(path))
            has_json = glob.glob(f"{path}/*.json")
            annname = str(path).split('/')[-1]
            returned_options[index] = {'choice': index, 'path': str(path), 'has_json': has_json, 'annname':annname }
    return str(options)

@mcp.tool()
def create_json_files(directory_selection: int) -> str:
    """
    Takes in a selection from the user and has a service create json files in the pdfs
    """
    ann_path = options[directory_selection]
    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    extraction = False
    if ann_pdf_files:
        for pdf_file in ann_pdf_files:
            extract_filter_pdf_to_json(pdf_file, ann_path)
            extraction = True
    return "An extraction has occured: " + str(extraction)

@mcp.tool()
def query_about_paper(load_selection: int, query: str) -> str:
    """Queries about a paper based on the selection made by the user."""
    global options

    # make selection is loaded into rag
    if load_selection > len(options):
        return "Invalid selection. Please select a valid index."

    option = options[load_selection]
    name = option.split('/')[-1]

    option = glob.glob(f"{option}/*.json") # Grabs all pdf doc json's
    

    if len(option) == 0:
        return "No JSON files found in the selected directory."
    init_engine(str(name), option[0])
    
    #response = query_llm(self.ann_config_name,prompt,pydantic_type_schema,max_chunks=20,token_budget=5000)
    # query the selection with querying the llm
    response = query_llm(name, query,ChatResponse)

    # return the response
    return response

@mcp.tool()
def query_graph_database(query: str) -> str:
    """ Queries a neo4j database that has information about neural networks """
    graph, session, driver = neo4j_connection()
    return get_answer_for_question(graph, query, driver)

@mcp.tool()
def create_ontology(directory_selection: int) -> str:
    """ Creates an ontology based on the selected directory as an integer index. """
    if directory_selection > len(options):
        return "Invalid selection. Please select a valid index."
    option = options[directory_selection]
    annname = option.split('/')[-1]
    output_ontology_filepath= os.path.join('data/user/', C.ONTOLOGY.USER_OWL_FILENAME)
    main(annname, option, use_user_owl=True, test_output_ontology_filepath = output_ontology_filepath )
    print('after main')
    driver = get_neo4j_connection()
    with driver.session() as session:

        rdf_format = "RDF/XML"
        importoptions = {
            "handleVocabUris": "SHORTEN",
            "typesToLabels": False,
            "keepLangTag": False,
            "handleMultival": "ARRAY"
        }
        
        # inline insert ontology -- this should be replaced with url insertion 
        content = open(output_ontology_filepath,'r').read() 
        counters = import_rdf(session, output_ontology_filepath, rdf_format, importoptions,inline=True,file_contents=content)

    return "Ontology created successfully."

@mcp.resource("ontology://appname")
def app_name() -> str:
    """
    Return the name of the application.
    """
    return "NCAT Server"




if __name__ == "__main__":
    if len(sys.argv) > 1:
        paperpath = sys.argv[1]
    
    # ths must be called -- in order to populate global variable options
    list_available_directories()
    #list_available_directories()
    #print(query_about_ann_config(1, "What is the title of this paper?"))
    #print(list_available_titles())
    # TODO: so I need to instaniate a name with the json paths
    
    #init_engine(self.ann_config_name, j)
    
    #response = query_llm(self.ann_config_name,prompt,pydantic_type_schema,max_chunks=20,token_budget=5000)
               

    #print(create_json_files(0))
    # Run the MCP server.
    mcp.run()
