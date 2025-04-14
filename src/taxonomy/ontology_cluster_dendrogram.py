from owlready2 import *
import numpy as np
from collections import deque
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from agg_to_decisiontree import run_agglomerative_tree_pipeline 
from utils.owl_utils import *

def get_most_specific_classes(instance):
    """
    Identify the most specific classes for an instance (classes with no subclasses among its types)
    
    Returns list of most specific classes
    """
    types = instance.is_a  # All classes the instance belongs to (direct and inferred)
    specific_types = [c for c in types if not any(d in c.subclasses() for d in types if d != c)]
    return specific_types

def get_graph_classes(ann_config, onto):
    """
    Perform BFS traversal of the ANNConfiguration's graph and collect specific classes of linked instances
    
    Returns a set of specific classes in the graph
    """
    visited = set()
    queue = deque([ann_config])
    classes = set()
    
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        
        if current != ann_config:
            specific_classes = get_most_specific_classes(current)
            classes.update(specific_classes)
        
        # Follow object properties to neighbors
        for prop in onto.object_properties():
            neighbors = prop[current]
            if isinstance(neighbors, list):
                queue.extend(neighbors)
            else:
                queue.append(neighbors)
    
    return classes
def get_sum_units(ann_config,onto):
    print(ann_config)
    networks = get_instance_property_values(ann_config, onto.hasNetwork.name)
    print(networks)
    valuesum = 0
    for network in networks:
        layers = get_instance_property_values(network, onto.hasLayer.name)
        for layer in layers:
            for prop in layer.get_properties():
                for value in prop[layer]:
                    if type(value) == int:
                        valuesum += value       
    print(valuesum)
    return valuesum

def create_taxonomy_from_owl(owl_file):

    onto = get_ontology(owl_file).load()
    
    ann_configs = list(onto.search(type=onto.ANNConfiguration))
    if not ann_configs:
        return
    
    print(f"Found {len(ann_configs)} ANNConfiguration instances: {[c.name for c in ann_configs]}")
    
    # Collect all unique specific classes across all graphs
    all_classes = set()
    sums = []
    for ann_config in ann_configs:
        sums.append([get_sum_units(ann_config,onto)])
        graph_classes = get_graph_classes(ann_config, onto)
        all_classes.update(graph_classes)
    
    all_classes = list(all_classes)
    print(f"Found {len(all_classes)} unique specific classes: {[c.name for c in all_classes]}")
    # Create binary feature vectors
    feature_vectors = []
    for ann_config in ann_configs:
        graph_classes = get_graph_classes(ann_config, onto)
        vector = [1 if c in graph_classes else 0 for c in all_classes]
        feature_vectors.append(vector)
   

    run_agglomerative_tree_pipeline(
        feature_vectors,
        np.array(sums),
        y=[str(ann_config) for ann_config in ann_configs],
        level_from_top=2,
        n_pca_components=20,
        plot_dendrogram=True,
        plot_pca=True,
        print_tree_rules=True,
        random_state=42)

    X = np.array(feature_vectors)
    
    Z = linkage(X, method='ward')  # 'ward' minimizes variance within clusters
    
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        labels=[str(ann_config) for ann_config in ann_configs],
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title("Taxonomy of ANNConfigurations")
    plt.xlabel("ANNConfigurations")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    plt.savefig('plot.png')

if __name__ == "__main__":
    owl_file_path = "data/owl/annett-o-test.owl"
    create_taxonomy_from_owl(owl_file_path)
