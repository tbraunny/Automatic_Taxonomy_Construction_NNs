import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import owlready2


def makeHierarchy(filename, linkage_matrix, names, classes={}):

    print(linkage_matrix)
    # Create an ontology
    onto = owlready2.get_ontology("http://example.org/clustering.owl")

    with onto:
        from owlready2 import Thing, ObjectProperty

        # 3.1 Define a class for cluster nodes
        class ClusterNode(Thing):
            pass

        # 3.2 Define an object property to link parent to children
        class has_subcluster(ObjectProperty):
            domain = [ClusterNode]
            range  = [ClusterNode]

    # Create individuals for leaf nodes
    num_leaves = linkage_matrix.shape[0] + 1
    leaf_nodes = []
    with onto:
        for i in range(num_leaves):
            leaf_node = ClusterNode(f"{names[i]}")
            leaf_nodes.append(leaf_node)

    # Build up internal cluster nodes from the linkage matrix
    current_index_to_owl_node = { i: leaf_nodes[i] for i in range(num_leaves) }

    with onto:
        for merge_index, (idx1, idx2, dist, sample_count) in enumerate(linkage_matrix):
            new_cluster_id = num_leaves + merge_index
            new_cluster = ClusterNode(f"Cluster_{new_cluster_id}")
            
            child1 = current_index_to_owl_node[idx1]
            child2 = current_index_to_owl_node[idx2]

            new_cluster.has_subcluster.append(child1)
            new_cluster.has_subcluster.append(child2)

            # You could also store distance or sample_count in a data property
            new_cluster.comment.append(
                f"Distance={dist}, SampleCount={int(sample_count)}"
            )

            current_index_to_owl_node[new_cluster_id] = new_cluster

    # (Optional) identify or label the root
    root_id = num_leaves + (num_leaves - 2)
    root_node = current_index_to_owl_node[root_id]

    print("Root node:", root_node)  
    print("Subclusters of root:", root_node.has_subcluster)
    onto.save(filename)

def compute_similarity(data):
    count = data.shape[0]
    distance_matrix = np.zeros((count,count))
    for i in range(count):
        for j in range(count):
            distance_matrix[i][j] = sum(abs(data[i] - data[j]))
    return distance_matrix

if __name__ == '__main__':

    data = np.array([
        [1.0, 2.0],
        [1.2, 2.1],
        [8.0, 8.2],
        [7.9, 8.0],
        [3.5, 4.0],
    ])

    distance_matrix = compute_similarity(data)
    linkage_matrix_nn = linkage(distance_matrix, method='average')
    makeHierarchy('out.owl',linkage_matrix_nn,['model1','model2','model3','model4','model5'])
