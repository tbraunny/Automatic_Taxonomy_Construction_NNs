import networkx as nx
import matplotlib.pyplot as plt
import io

# Create a graph
#G = nx.Graph()
def visualizeTaxonomy(graphml,isString=True):
    if isString:
        handle = io.StringIO(graphml)
        G = nx.read_graphml(handle)
    else:
        G = nx.read_graphml(graphml) # must be a file path

    pos = nx.spring_layout(G)  # Positions for nodes

    # Draw the graph
    node_colors = [G.nodes[n].get("color", "black") for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=1000, font_size=10)

    # Draw node labels (annotations)
    node_labels = {n: G.nodes[n]["annConfigs"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="black")

    # Draw edge labels (annotations
    print(G.edges)
    for u, v in G.edges:
        print(G.edges[u,v])
    edge_labels = {(u, v): (G.nodes[u]['splitKey'],G.nodes[v]['splitKey']) for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the plot
    plt.show()

