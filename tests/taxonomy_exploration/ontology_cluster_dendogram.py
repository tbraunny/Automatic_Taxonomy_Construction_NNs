from owlready2 import *
import numpy as np
from collections import deque
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def get_graph_classes(ann_config, onto):
    """
    Perform BFS traversal of the ANNConfiguration's graph and collect schema classes of linked instances
    
    Returns a set of schema classes in the graph
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
            schema_classes = current.is_a # Get schema classes of the instance
            classes.update(schema_classes)
        
        # Follow object properties to neighbors
        for prop in onto.object_properties():
            neighbors = prop[current]
            if isinstance(neighbors, list):
                queue.extend(neighbors)
            else:
                queue.append(neighbors)
    
    return classes

def create_taxonomy_from_owl(owl_file):

    onto = get_ontology(owl_file).load()
    
    ann_configs = list(onto.search(type=onto.ANNConfiguration))
    if not ann_configs:
        return
    
    print(f"Found {len(ann_configs)} ANNConfiguration instances: {[c.name for c in ann_configs]}")
    
    # Collect all unique schema classes across all annconfig instance graphs
    all_classes = set()
    for ann_config in ann_configs:
        graph_classes = get_graph_classes(ann_config, onto)
        all_classes.update(graph_classes)
    
    all_classes = list(all_classes)
    print(f"Found {len(all_classes)} unique schema classes: {[c.name for c in all_classes]}")
    
    # Create binary feature vectors
    feature_vectors = []
    for ann_config in ann_configs:
        graph_classes = get_graph_classes(ann_config, onto)
        vector = [1 if c in graph_classes else 0 for c in all_classes]
        feature_vectors.append(vector)
    
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