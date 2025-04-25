import rdflib
import subprocess
from mcp.server.fastmcp import FastMCP
import glob
import json
import owlready2
from owlready2 import *
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from rdflib import Graph, URIRef

import time

#from ... import *
#from taxonomy import llm_service,criteria,create_taxonomy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.taxonomy import llm_service,create_taxonomy

# Create an MCP server named "OWL Server
mcp = FastMCP("OWL Server")

# Load the ontology using rdflib.
# Adjust "ontology.owl" and its format as needed.
g = rdflib.Graph()
g.parse("./data/owl/annett-o-0.1.owl", format="xml")


onto = get_ontology("./data/owl/annett-o-0.1.owl").load()

options = glob.glob('./data/owl/*.owl')


def extract_subgraph(graph: Graph, root_iri: str) -> Graph:
    """
    Extract all triples reachable from a given root IRI (like a BFS).
    This function returns a sub-graph with those triples.
    """
    subg = Graph()
    
    root = URIRef(root_iri)
    # Use a BFS/DFS approach on the "networkx" version of the graph
    nxg = rdflib_to_networkx_graph(graph)
    
    # BFS from the root
    visited = set()
    queue = [root]
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        # Add all triples where current is subject
        for p, o in graph.predicate_objects(subject=current):
            subg.add((current, p, o))
            if o not in visited and (o in nxg):
                queue.append(o)
        # Also add all triples where current is object
        for s, p in graph.subject_predicates(object=current):
            subg.add((s, p, current))
            if s not in visited and (s in nxg):
                queue.append(s)
    return subg




#options = glob.glob('./ontology/*.owl')
#options = { index : option for index, option in enumerate(options)}
#print(options)

# this might be better as a resource
@mcp.tool()
def list_available_options() -> str:
    """Lists available options that can be loaded that has option to file."""
    options = glob.glob('./ontology/*.owl')
    options = { index : option for index, option in enumerate(options)}
    return str(options)

@mcp.tool()
def load_selection(option: int) -> bool:
    """Loads a selection by providing a integer that loads a file"""
    try:
        g = rdflib.Graph()
        g.parse(options[option], format="xml")
        onto = get_ontology(options[option]).load()
        return True
    except Exception as e:
        print(f"Error invalid option: {e}")
        return False

@mcp.tool()
def list_neural_networks():
    """Lists all neural networks in present ontology. Returns a json dictionary as {index : neural network} """
    annconfigurations = onto.ANNConfiguration.instances()
    #annconfigurations = list(annconfigurations)
    annconfigurations = { index:annconfig for index, annconfig in enumerate(annconfigurations)}
    return str(annconfigurations)




@mcp.tool()
def get_neural_network(option: int) -> str:
    annconfigurations = list(onto.ANNConfiguration.instances())
    #annconfigurations = list(annconfigurations)
    annconfigs = { index:annconfig for index, annconfig in enumerate(annconfigurations)}

    selection = annconfigs[option].iri
    # extract subgraph of the selection annconfig and return json
    option = extract_subgraph(g,selection)
    jsonld_data = option.serialize(format="json-ld", indent=2)
    return jsonld_data

#print(get_neural_network(0))

@mcp.tool()
def sparql_query(query: str) -> str:
    """Execute a SPARQL query over the loaded ontology and return the results as text."""
    try:
        results = g.query(query)
        output_lines = []
        for row in results:
            print(row,"-------------------")
            output_lines.append(", ".join(str(item) for item in row))
        return "\n".join(output_lines) if output_lines else "No results found."
    except Exception as e:
        #print('Error executing SPARQL query:', e)
        return f"Error executing SPARQL query: {e}"

@mcp.tool()
def get_classes() -> str:
    """
    Return a list of classes in the ontology.
    """
    classes = set()
    for s, p, o in g.triples((None, rdflib.RDF.type, None)):
        classes.add(str(o))
        #break
    return ",".join(classes) if classes else "No classes found."

@mcp.tool()
def create_taxonomy_tool(query: str) -> str:
    """
    Takes in a sentence from the user that the llm clarifies and a faceted taxonomy is returned
    """
    try:
        oc = llm_service.llm_create_taxonomy(query, onto)
        taxonomy_creator = create_taxonomy.TaxonomyCreator(  onto,criteria=oc.criteriagroup)
        format='json'
        topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(format=format,faceted=True)
        #output = 'this is a test'
    except Exception as e: 
        return f"Error creating taxonomy: {e}"
        
    return str(output)

@mcp.resource("ontology://info")
def ontology_info() -> str:
    """
    Return basic information about the loaded ontology (e.g., number of triples loaded).
    """
    return f"Ontology loaded from 'ontology.owl' contains {len(g)} triples."

@mcp.resource("ontology://appname")
def app_name() -> str:
    """
    Return the name of the application.
    """
    return "OWL Server"

if __name__ == "__main__":
    start_time = time.time()
    #print('testing')
    #print(create_taxonomy_tool('create a taxonomy of neural network layers'))
    #print('after')
    #print(time.time() - start_time)
    # Run the MCP server.
    mcp.run()

