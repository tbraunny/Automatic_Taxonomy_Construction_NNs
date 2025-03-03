from rdflib import Namespace, URIRef
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import ResultRow
from utils.constants import Constants as C
from typing import List

# Function to parse individual SPARQL rows
def parse_SPARQLRow(sparql_row: ResultRow, namespace: Namespace) -> str:
    """
    Parses an individual SPARQLResult row and extracts the component as a string.

    Args:
        sparql_row: The row containing one or more RDF terms.
        namespace: The namespace defined for the ontology

    Returns:
        str: Parsed component as a string, with the namespace removed if applicable.
    """
    parsed_row = [
        str(term).replace(namespace, '') if isinstance(term, URIRef) else str(term)
        for term in sparql_row
    ]
    return parsed_row

# Function to parse entire SPARQLResult
def parse_SPARQLResult(sparql_result: SPARQLResult, namespace: Namespace) -> List[List[str]]:
    """
    Parses a SPARQLResult object and extracts each component in each row as a list of strings.

    Args:
        sparql_result (SPARQLResult): The SPARQL query result to parse.
        namespace: The namespace defined for the ontology

    Returns:
        List[List[str]]: A list of parsed result rows, where each row is a list of strings.
    """
    return [parse_SPARQLRow(sparql_row=row, namespace=namespace) for row in sparql_result]