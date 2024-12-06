from owlready2 import get_ontology
import networkx as nx
from pyvis.network import Network
import os
import sys
import json

# Load constants from utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
from constants import Constants as C  # Import Constants from utils

# Helper function to load ontology
def load_ontology(filepath):
    print(f"Loading ontology from {filepath}...")
    return get_ontology(filepath).load()

# Helper function to clean instance names
def clean_instance_name(instance):
    return str(instance).split(".")[-1]

# Recursively build a dictionary of relationships for an instance
def build_instance_relations(instance, visited=None, depth=3):
    """
    Recursively build a dictionary of relationships for an instance.

    Args:
        instance: The root instance to start from.
        visited: A set of visited instances to avoid cycles.
        depth: The depth of recursion (default: 3).

    Returns:
        A dictionary representing the relationships for this instance.
    """
    if visited is None:
        visited = set()
    if depth == 0 or instance in visited:
        return {}

    visited.add(instance)
    relations_dict = {clean_instance_name(instance): {}}  # Start with the current instance

    for prop in instance.get_properties():
        try:
            values = list(prop[instance])  # Get the property values
            if values:
                relations_dict[clean_instance_name(instance)][prop.name] = [
                    clean_instance_name(value) for value in values
                ]
                for value in values:
                    # Recursively build relationships for each related instance
                    sub_relations = build_instance_relations(value, visited, depth - 1)
                    relations_dict.update(sub_relations)  # Merge sub-relations into the main dictionary
        except AttributeError:
            continue

    return relations_dict

# Convert relationships dictionary into a graph
def build_graph_from_relations(relations_dict):
    """
    Build a NetworkX graph from a relationships dictionary.

    Args:
        relations_dict: The dictionary of relationships.

    Returns:
        A NetworkX graph.
    """
    G = nx.DiGraph()

    for instance, props in relations_dict.items():
        G.add_node(instance, label="Instance")  # Add the instance as a node
        for prop, targets in props.items():
            for target in targets:
                G.add_node(target, label="Instance")
                G.add_edge(instance, target, label=prop)  # Add edges with property names as labels

    return G


def visualize_with_pyvis(graph, output_filename="graph.html"):
    """
    Visualize the graph using PyVis and save it as an HTML file.

    Args:
        graph: A NetworkX graph object.
        output_filename: The name of the output HTML file.
    """
    net = Network(
        height="1500px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000",
    )
    net.barnes_hut()

    # Add nodes with improved styling
    for node in graph.nodes:
        net.add_node(
            node,
            title=f"Details about {node}",
            label=node,
            color="#f74d6c" if node == list(graph.nodes)[0] else "#61bfea",
            # shape="circle",
            size=20,
        )

    # Add edges with improved styling
    for edge in graph.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            title=edge[2].get('label', ''),
            label=edge[2].get('label', ''),
            color="#a8a8a8",
            width=1.5,
            arrows="to",
        )

    # Adjust physics and layout for better spacing
    net.toggle_physics(True)
    net.set_options(json.dumps({
        "nodes": {
            "font": {"size": 14, "color": "#343434"},
            "scaling": {"min": 15, "max": 30},
            "shapeProperties": {"useBorderWithImage": True}
        },
        "edges": {
            "color": {"color": "#a8a8a8", "highlight": "#848484"},
            "smooth": {"type": "dynamic"}
        },
        "physics": {
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 1000, "fit": True},
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "springLength": 300,  # Increase spring length for better spacing
                "springConstant": 0.05
            },
            "solver": "forceAtlas2Based"
        },
        "layout": {
            "improvedLayout": True  # Optimize layout for readability
        }
    }))

    # Save the graph
    net.save_graph(output_filename)
    print(f"Graph saved to {output_filename}")


# Main function to test recursive instance graph
def process_ontology():
    # Load the ontology
    print("called")
    ontology_file = os.path.join(ROOT_DIR, "data/owl", C.ONTOLOGY.FILENAME)
    ontology = load_ontology(ontology_file)

    # Get an instance of ANNConfiguration
    class_name = "ANNConfiguration"
    cls = ontology[class_name]
    if not cls:
        raise ValueError(f"Class '{class_name}' not found in the ontology.")

    instances = list(cls.instances())
    if not instances:
        print(f"No instances found for class '{class_name}'.")
        return

    # Test with the first instance
    root_instance = instances[0]
    print(f"Generating graph for instance: {clean_instance_name(root_instance)}")

    # Build relationships recursively
    relations_dict = build_instance_relations(root_instance, depth=2)

    # Convert the relationships dictionary to a graph
    graph = build_graph_from_relations(relations_dict)

    # Visualize the graph
    output_path = os.path.join(ROOT_DIR, "front_end/static", "instance_graph.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_with_pyvis(graph, output_path)

if __name__ == "__main__":
    process_ontology()
