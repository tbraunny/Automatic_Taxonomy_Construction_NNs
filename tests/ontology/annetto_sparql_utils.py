from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import ResultRow
from typing import List

# Initialize the graph
g = Graph()
g.parse("/Users/josueochoa/Desktop/Capstone/Automatic_Taxonomy_Construction_NNs/owl_parsing/annett-o-0.1.owl", format="xml")  # Replace with the path to your OWL file

# Define the namespace
ns = Namespace("http://w3id.org/annett-o/")

# Function to parse individual SPARQL rows
def parse_SPARQLRow(sparql_row: ResultRow) -> str:
    """
    Parses an individual SPARQLResult row and extracts the component as a string.

    Args:
        sparql_row: The row containing one or more RDF terms.

    Returns:
        str: Parsed component as a string, with the namespace removed if applicable.
    """
    parsed_row = [
        str(term).replace(ns, '') if isinstance(term, URIRef) else str(term)
        for term in sparql_row
    ]
    return parsed_row

# Function to parse entire SPARQLResult
def parse_SPARQLResult(sparql_result: SPARQLResult) -> List[List[str]]:
    """
    Parses a SPARQLResult object and extracts each component in each row as a list of strings.

    Args:
        sparql_result (SPARQLResult): The SPARQL query result to parse.

    Returns:
        List[List[str]]: A list of parsed result rows, where each row is a list of strings.
    """
    return [parse_SPARQLRow(row) for row in sparql_result]

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
    parsed_ann_configuration = parse_SPARQLRow(ann_config_row)
    parsed_networks = parse_SPARQLResult(ann_network_results)
    
    print(f"Networks for {parsed_ann_configuration}: {parsed_networks}")