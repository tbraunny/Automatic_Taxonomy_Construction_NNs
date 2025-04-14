import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
import ace_tools_open as tools

def run_agglomerative_tree_pipeline(
    X,
    otherdataset,
    y=None,
    level_from_top=2,
        n_pca_components=50,
    plot_dendrogram=True,
    plot_pca=True,
    print_tree_rules=True,
    random_state=42
):
    """
    Runs a pipeline that:
      1. Reduces dimensionality of X with PCA.
      2. Computes hierarchical clustering (Ward linkage) on the PCA-transformed data.
      3. Cuts the dendrogram at a level specified by level_from_top (e.g. level 2 means one merge below the root).
      4. Trains a decision tree to predict the cluster labels from the PCA features.
      5. Optionally prints the decision tree rules and cluster-to-true-label mapping (if y is provided).
      6. Optionally visualizes the dendrogram and PCA scatter plot.
    
    Parameters:
      X (array-like): Input feature data (n_samples x n_features).
      y (array-like, optional): True labels for the samples. If provided, cluster-to-label mapping is shown.
      level_from_top (int): Which level (from the top) to use for cutting the dendrogram.
      n_pca_components (int): Number of principal components for PCA reduction.
      plot_dendrogram (bool): If True, plots the dendrogram with the chosen cut level.
      plot_pca (bool): If True, plots the PCA-reduced data colored by cluster label.
      print_tree_rules (bool): If True, prints the decision tree rules.
      random_state (int): Random seed (currently used if any randomness is needed in further modifications).
    
    Returns:
      results (dict): Contains the PCA-transformed data, cluster labels, the trained decision tree,
                      and (if provided) the cluster-to-label mapping DataFrame.
    """
    # Step 1: PCA Dimensionality Reduction
    #pca = PCA(n_components=n_pca_components)
    #X_pca = pca.fit_transform(X)
    
    # Step 2: Compute Hierarchical Clustering (Ward linkage)
    X = np.array(X)
    Z = linkage(X, method='ward')
    n_samples = X.shape[0]
    
    # Build a dictionary representing the tree from the linkage matrix.
    tree = {}
    for i, row in enumerate(Z):
        idx1, idx2 = int(row[0]), int(row[1])
        tree[n_samples + i] = (idx1, idx2)
    
    def compute_depths(Z, n):
        depths = {i: 0 for i in range(n)}
        for i, row in enumerate(Z):
            idx1, idx2 = int(row[0]), int(row[1])
            d = max(depths[idx1], depths[idx2]) + 1
            depths[n + i] = d
        return depths
    
    depths = compute_depths(Z, n_samples)
    max_depth = max(depths.values())
    print("Maximum tree depth:", max_depth)
    
    # Determine cut depth (from leaves) corresponding to level_from_top.
    cut_depth = max_depth - (level_from_top - 1)
    print(f"Cutting at depth (from leaves) = {cut_depth} corresponding to level {level_from_top} from the top.")
    
    # Step 3: Extract clusters from the tree using a recursive function.
    def get_clusters(node_index, tree, depths, cut_depth):
        if node_index < n_samples:
            return [node_index]
        if depths[node_index] == cut_depth:
            return [node_index]
        left, right = tree[node_index]
        return get_clusters(left, tree, depths, cut_depth) + get_clusters(right, tree, depths, cut_depth)
    
    root = n_samples + Z.shape[0] - 1
    cluster_nodes = get_clusters(root, tree, depths, cut_depth)
    print("Number of clusters at chosen level:", len(cluster_nodes))
    
    # Function to get leaves of a node.
    def get_leaves(node_index, tree):
        if node_index < n_samples:
            return [node_index]
        else:
            left, right = tree[node_index]
            return get_leaves(left, tree) + get_leaves(right, tree)
    
    # Map each sample to a cluster label.
    cluster_labels = np.empty(n_samples, dtype=int)
    for label, node in enumerate(cluster_nodes):
        leaves = get_leaves(node, tree)
        for leaf in leaves:
            cluster_labels[leaf] = label
    
    # Step 4: Train Decision Tree to predict cluster labels from PCA features.
    tree_clf = DecisionTreeClassifier(random_state=random_state)
    tree_clf.fit(otherdataset, cluster_labels)
    pred_cluster_labels = tree_clf.predict(otherdataset)
    
    acc_tree_vs_clusters = accuracy_score(cluster_labels, pred_cluster_labels)
    print("Decision Tree vs. Clustering Labels Accuracy:", acc_tree_vs_clusters)
    
    if print_tree_rules:
        # Use export_text to print a human-readable decision tree.
        feature_names = [f"Feature{i+1}" for i in range(otherdataset.shape[1])]
        tree_rules = export_text(tree_clf, feature_names=feature_names)
        print("\nDecision Tree Rules:\n")
        print(tree_rules)
    
    # Optionally, print a mapping from clusters to true labels if y is provided.
    mapping_df = None
    if y is not None:
        df = pd.DataFrame({"True Label": y, "Cluster Label": cluster_labels})
        mapping_df = df.groupby("Cluster Label")["True Label"].value_counts().unstack(fill_value=0)
        print("\nMapping of clusters to true labels:")
        print(mapping_df)
    
    # Step 5: Visualization
    if plot_dendrogram:
        plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='level', p=5, labels=y)
        # Determine cut line from the cluster nodes.
        cut_distances = []
        for node in cluster_nodes:
            if node >= n_samples:
                idx = node - n_samples
                cut_distances.append(Z[idx, 2])
        cut_line = min(cut_distances) if cut_distances else 0
        plt.axhline(y=cut_line, color='r', linestyle='--', label=f'Cut at level {level_from_top}')
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index or (Cluster Size)")
        plt.ylabel("Distance")
        plt.legend()
        plt.show()
    
    if plot_pca:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=100)
        plt.title(f"PCA-Reduced Data with Clusters from Level {level_from_top}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        plt.show()
    
    results = {
        "X_pca": X,
        "cluster_labels": cluster_labels,
        "decision_tree": tree_clf,
        "cluster_mapping": mapping_df,
        "decision_tree_rules": tree_rules if print_tree_rules else None
    }
    return results


