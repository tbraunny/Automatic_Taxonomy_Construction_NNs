from rdflib import Graph, Namespace
from utils.owl.sparql import parse_SPARQLResult, parse_SPARQLRow
from utils.constants import Constants as C
from typing import List

# Initialize the graph
g = Graph()
g.parse(f"./data/owl/{C.ONTOLOGY.FILENAME}", format="xml")  # Replace with the path to your OWL file

# Define the namespace
ns = Namespace(C.ONTOLOGY.NAMESPACE)

# SPARQL query to get all ANNConfiguration instances
ann_config_query = f"""
                    SELECT ?instance
                    WHERE {{
                        ?instance rdf:type <{ns}ANNConfiguration> .
                    }}
                    """

ann_config_results = g.query(ann_config_query)
# ann_config_results ResultRows in form:
# (rdflib.term.URIRef('http://w3id.org/annett-o/AAE'),)
# (rdflib.term.URIRef('http://w3id.org/annett-o/GAN'),)
# (rdflib.term.URIRef('http://w3id.org/annett-o/simple_classification'),)

for ann_config_row in ann_config_results:
    ann_configuration = ann_config_row[0]

    # Query to get all networks associated with ann_configuration
    ann_network_query = f"""
                    SELECT ?network
                    WHERE {{
                        <{ann_configuration}> <{ns}hasNetwork> ?network .
                    }}
                    """
    ann_network_results = g.query(ann_network_query)
    
    # Use parse_SPARQLRow for individual row parsing or parse_SPARQLResult for all results
    parsed_ann_configuration = parse_SPARQLRow(ann_config_row, namespace=ns)
    parsed_networks = parse_SPARQLResult(ann_network_results, namespace=ns)
    
    print(f"Networks for {parsed_ann_configuration}: {parsed_networks}")