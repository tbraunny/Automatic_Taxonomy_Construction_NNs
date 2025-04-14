from owlready2 import get_ontology
import networkx as nx
from pyvis.network import Network
import os
import json
import seaborn as sns
import numpy as np
import hashlib
from utils.constants import Constants as C

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# class color map to store dynamically assigned colors
CLASS_COLOR_MAP = {}

# seed for consistent colors
SEED = 42
np.random.seed(SEED)

# toggle to dump json output of relationships
DUMP_JSON = False

def generate_dynamic_color_palette(n_classes):
    # generate a dynamic palette with n visually distinct colors
    palette = sns.color_palette("inferno", n_classes)
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in palette]

def generate_color_from_hash(name, palette_size=20):
    """
    Generate a deterministic color index based on a stable hash of the name.
    Args:
        name (str): The name to hash.
        palette_size (int): The number of available colors in the palette.
    Returns:
        int: The index of the color in the palette.
    """
    # Use a stable hash function (SHA-256) and take the first few bytes
    stable_hash = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)
    return stable_hash % palette_size

def get_color_for_class(instance):
    """
    Assign a unique color to a node based on its class type using a hash function.
    """
    class_type = instance.is_a[0].name if hasattr(instance, "is_a") and instance.is_a else "Unknown"

    if class_type not in CLASS_COLOR_MAP:
        # Use the hash-based color selection
        palette = generate_dynamic_color_palette(20)  # Generate a palette of 20 colors
        color_index = generate_color_from_hash(class_type, len(palette))
        CLASS_COLOR_MAP[class_type] = palette[color_index]

    return CLASS_COLOR_MAP[class_type]

def load_ontology(filepath):
    # load the ontology from the given file path
    print(f"loading ontology from {filepath}...")
    return get_ontology(filepath).load()

def clean_instance_name(instance):
    # extract and return the cleaned name of an instance
    return str(instance).split(".")[-1]

def build_instance_relations(instance, visited=None, depth=3):
    # recursively build a nested dictionary of relationships for an instance
    if visited is None:
        visited = set()
    if depth == 0 or instance in visited:
        return {}
    visited.add(instance)
    instance_name = clean_instance_name(instance)
    relations_dict = {
        instance_name: {
            "class": instance.is_a[0].name if hasattr(instance, "is_a") and instance.is_a else "Unknown",
            "instance": instance,
            "properties": {}
        }
    }
    for prop in instance.get_properties():
        try:
            values = list(prop[instance])
            if values:
                prop_name = prop.name
                relations_dict[instance_name]["properties"][prop_name] = []
                for value in values:
                    if hasattr(value, "is_a"):  # handle owl instances
                        sub_relations = build_instance_relations(value, visited, depth - 1)
                        if sub_relations:
                            relations_dict[instance_name]["properties"][prop_name].append(sub_relations)
                    else:  # handle literals or non-owl objects
                        relations_dict[instance_name]["properties"][prop_name].append(str(value))
        except Exception as e:
            print(f"error processing property '{prop.name}': {e}")
            continue
    return relations_dict

def sanitize_relations_dict(relations_dict):
    # remove non-serializable objects (e.g., owl instances) from the dictionary
    sanitized = {}
    for instance_name, data in relations_dict.items():
        sanitized[instance_name] = {
            "class": data["class"],
            "properties": {}
        }
        for prop, targets in data["properties"].items():
            sanitized[instance_name]["properties"][prop] = [
                sanitize_relations_dict(target) if isinstance(target, dict) else target for target in targets
            ]
    return sanitized

def build_graph_from_relations(relations_dict, graph=None, root_node=None):
    # build a graph from the relations dictionary
    if graph is None:
        graph = nx.DiGraph()
    if root_node is None:
        root_node = next(iter(relations_dict))
    for instance_name, data in relations_dict.items():
        owl_instance = data["instance"]
        graph.add_node(
            instance_name,
            label=data["class"],
            color='#000000' if instance_name == root_node else get_color_for_class(owl_instance),
            size=40 if instance_name == root_node else 20,
        )
        for prop, targets in data.get("properties", {}).items():
            for target in targets:
                if isinstance(target, dict):
                    target_name = list(target.keys())[0]
                    target_data = target[target_name]
                    target_instance = target_data["instance"]
                    graph.add_node(
                        target_name,
                        label=target_data["class"],
                        color=get_color_for_class(target_instance),
                        size=20
                    )
                    graph.add_edge(instance_name, target_name, label=prop)
                    build_graph_from_relations(target, graph, root_node)
    return graph

def visualize_with_pyvis(graph, output_filename="graph.html"):
    # visualize the graph using pyvis
    net = Network(height="1500px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
    net.barnes_hut()
    for node, data in graph.nodes(data=True):
        net.add_node(
            node,
            title=f"Class: {data['label']}",
            label=node,
            color=data["color"],
            size=data["size"],
        )
    for edge in graph.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            title=edge[2].get("label", ""),
            label=edge[2].get("label", ""),
            color="#a8a8a8",
            width=1.5,
            arrows="to",
        )
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
                "springLength": 300,
                "springConstant": 0.05
            },
            "solver": "forceAtlas2Based"
        },
        "layout": {
            "improvedLayout": True,
            "hierarchical": False
        }
    }))
    net.save_graph(output_filename)
    print(f"graph saved to {output_filename}")

def process_ontology():
    print("called")
    # load the ontology
    ontology_file = os.path.join(ROOT_DIR, "data/owl", C.ONTOLOGY.FILENAME)
    ontology = load_ontology(ontology_file)
    # get an instance of the root class
    class_name = "ANNConfiguration"
    cls = ontology[class_name]
    if not cls:
        raise ValueError(f"class '{class_name}' not found in the ontology.")
    instances = list(cls.instances())
    if not instances:
        print(f"no instances found for class '{class_name}'.")
        return
    # test with the first instance
    root_instance = instances[0]
    print(f"generating graph for instance: {clean_instance_name(root_instance)}")
    # build relationships recursively
    relations_dict = build_instance_relations(root_instance, depth=3)
    if DUMP_JSON:
        sanitized_dict = sanitize_relations_dict(relations_dict)
        print(json.dumps(sanitized_dict, indent=2))
    # convert the relationships dictionary to a graph
    graph = build_graph_from_relations(relations_dict)


    # Visualize the graph
    output_path = os.path.join(ROOT_DIR, "front_end/static", "instance_graph.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_with_pyvis(graph, output_path)


if __name__ == "__main__":
    process_ontology()
