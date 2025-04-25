import rdflib
import networkx as nx

from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib.namespace import RDFS
from rdflib import URIRef,Graph, RDF

import pandas as pd


def node_subst_cost(node1, node2):
    '''
    A node substition function for network that computes the similarity between two atributes of nodes during substition.
    '''
    type1 = node1.get('http://www.w3.org/1999/02/22-rdf-syntax-ns#type',"") 
    type2 = node2.get('http://www.w3.org/1999/02/22-rdf-syntax-ns#type',"")
    return 0 if type1 == type2 and type1 != "" and type2 != "" else 1

# this will grab everything under a node -- this aimmed toward models and neural networks in fair nets 
def get_all_nodes(graph: Graph, node: URIRef) -> list:
    '''
    Gets all nodes associated to a node in the rdflib graph. 
    '''
    nodelist = []
    # iterating only two layers -- this code could be made recursive
    for predicate, obj in graph.predicate_objects(subject=node):
        #print(predicate,obj)
        nodelist.append(obj)
        for predicate, obj in graph.predicate_objects(subject=obj):
            nodelist.append(obj)
            #print(predicate,obj)
    return nodelist

def get_node_properties(graph: Graph, node: URIRef) -> dict:
    """
    Retrieve all properties and their associated values for a given node (URIRef) in an RDF graph.

    Parameters:
        graph (rdflib.Graph): The RDF graph to query.
        node (rdflib.URIRef): The URI of the node whose properties are to be retrieved.

    Returns:
        dict: A dictionary where keys are property URIs and values are lists of associated objects.
    """
    properties = {}
    for predicate, obj in graph.predicate_objects(subject=node):
        prop_uri = str(predicate)
        if prop_uri not in properties:
            properties[prop_uri] = []
        properties[prop_uri].append(obj)
        #node[prop_uri] = obj
    return properties


def compareGraphs(nx_graph:  nx.MultiDiGraph, nodes1: list,nodes2: list):
    '''
    Compares two subgraphs from the passed in network x graph
    Parameters:
        nx_graph: The original graph
        nodes1: Nodes for the first subgraph
        nodes2: Nodes for the second subgraph
    '''
    G1 = nx_graph.subgraph(nodes1)
    G2 = nx_graph.subgraph(nodes2)
    ged_iter = nx.optimize_graph_edit_distance(
        G1, G2, node_subst_cost=node_subst_cost #, edge_subst_cost=edge_subst_cost
    )
    edit_distance = next(ged_iter)
    return edit_distance

if __name__ == '__main__':
    '''
    This method compare the networks inside of ann config, but does not tie back to the original config!
    '''

    # Create an RDF graph
    g = rdflib.Graph()
    
    # fairnets
    fairnets = False
    if fairnets:
        g.parse("fairnet.ttl", format="turtle")  # Replace with your RDF file and format

        #target_class = URIRef("https://w3id.org/nno/ontology#BaseModel")
        #target_class = URIRef("https://w3id.org/nno/ontology#NeuralNetwork")

        #layer_class = URIRef("https://w3id.org/nno/ontology#hasLayer")
        #instances = [s for s, p, o in g.triples((None, RDF.type, target_class))]

        models = []
        search_class = URIRef("https://w3id.org/nno/ontology#hasModel")
        target_class = URIRef("https://w3id.org/nno/ontology#Model")
        models = [s for s, p, o in g.triples((None, RDF.type, target_class))]

        #for instance in instances:
        #    break # this is slow!!!
        #    #print(instance)
        #    models += [(s,o) for s, p, o in g.triples((instance, search_class, None))]
        #    for model,out in models:
        #        layers = [(s,p,o) for s, p, o in g.triples((out, layer_class, None))]
    else:
        g.parse("annett-o-0.1.owl", format="xml")  # Replace with your RDF file and format
        models = []
        target_class = URIRef("http://w3id.org/annett-o/Network")
        models = [s for s, p, o in g.triples((None, RDF.type, target_class))]
        print(models)

    # Convert to NetworkX MultiDiGraph
    nx_graph = rdflib_to_networkx_multidigraph(g)

    # Now you can use NetworkX functions on the graph
    print(nx_graph.number_of_nodes())
    print(nx_graph.number_of_edges())

    # Populating properties for network -- this needs to be done ... otherwise we can't compare the nodes 
    for model in models:
        nodes = get_all_nodes(g,model)        
        for node in nodes:
            node_properties = get_node_properties(g, node)
            for key in node_properties:
                nx_graph.nodes[node][key] = node_properties[key]


    data = {'model1':[],'model2':[],'distance':[]}

    # iterating over all models found -- this could be change to AnnConfig or NeuralNetwork in fairnets 
    # TODO: This is slow. Will need to be made faster for fairnets.
    for model in models:
        nodes1 = get_all_nodes(g,model) 
        for model2 in models:
            nodes2 = get_all_nodes(g,model2)
            distance = compareGraphs(nx_graph,nodes1,nodes2)
            data['model1'].append(str(model)) 
            data['model2'].append(str(model2)) 
            data['distance'].append(distance) 
    

    # saving out the result to a distances csv
    pd.DataFrame(data).to_csv('distances.csv')
