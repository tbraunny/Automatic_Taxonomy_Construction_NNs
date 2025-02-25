import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import networkx as nx
from sklearn.decomposition import PCA
import cuml 
import graphviz
from processing import parse_networks,LAYER_MAPPING,REVERSE_MAPPING,create_chain_graph
from model import GraphAutoencoder
import plotly.graph_objects as go

st.title("Graph Autoencoder Visualization")
st.write("This app loads a pretrained Graph Autoencoder, computes the node encodings for a sample graph, and visualizes the 2D projection using a DOT (Graphviz) plot.")

show_networks = st.checkbox("Show network visualizations", False)

# Sample JSON string (you can replace this with your own JSON or upload file)
sample_json = open('./networks.json','r').read()
st.subheader("Parsing Networks")
parsed_networks = parse_networks(sample_json)
print(parsed_networks[2])
if not parsed_networks:
    st.error("No valid networks found in the JSON.")
else:
    st.write(f"Parsed {len(parsed_networks)} networks successfully.")

# Convert each parsed network into a chain graph.
dataset = [create_chain_graph(net_name, tokens) for net_name, tokens in parsed_networks]
names = [i[0] for i in dataset]
dataset = [i[1] for i in dataset]

selected = st.multiselect('Select from these variables:', names)
#showing = []
#for name in names:
#    showing.append(st.checkbox(name))


# Process all networks: run model to get node embeddings and collect edge info.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate (or load) the pretrained model.
# For demonstration, we simulate a pretrained model.
num_tokens = max(LAYER_MAPPING.values()) + 1
model = GraphAutoencoder(num_tokens=num_tokens, embed_dim=8, hidden_dim=16, latent_dim=8)
model = model.to(device)
model.eval()

all_embeddings_list = []
graph_boundaries = []  # Each element: dict with keys: name, start, end, edge_index.
current_index = 0


indices = []
for index, data_obj in enumerate(dataset):
    if not data_obj['name'] in selected:
        continue
    #print(data_obj.name)
    data_obj = data_obj.to(device)
    with torch.no_grad():
        z = model.encode(data_obj)  # [num_nodes, latent_dim]
    num_nodes = z.size(0)
    all_embeddings_list.append(z.cpu())
    graph_boundaries.append({
        'name': data_obj.name,
        'start': current_index,
        'end': current_index + num_nodes,
        'edge_index': data_obj.edge_index.cpu().numpy(),
        'embedding': z
    })
    current_index += num_nodes
    indices.append(index)

all_embeddings = torch.cat(all_embeddings_list, dim=0)  # [total_nodes, latent_dim]
pca = PCA(n_components=2)
#pca = cuml.UMAP(n_components=2)
all_embeddings_2d = pca.fit_transform(all_embeddings.numpy())

# Only show visualization if the checkbox is enabled.
if show_networks:
    # Prepare a color palette.
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    node_traces = []
    edge_traces = []
    
    for idx, graph_info in enumerate(graph_boundaries):
        idx = indices[idx]
        if not graph_info['name'] in selected:
            continue
        start, end = graph_info['start'], graph_info['end']
        coords = all_embeddings_2d[start:end]  # shape: [n, 2]
        # Retrieve original token values for labels.
        # Since each graph is created from the dataset list, we use the same index.
        #token_labels = [f"Token {int(t)}" for t in dataset[idx].x.cpu().numpy()]
        token_labels = [f"Token {REVERSE_MAPPING[int(t)]}" for t in dataset[idx].x.cpu().numpy()]
        print(token_labels)
        print(len(coords))
        print(start,end)
        print(dataset[idx].x)
        node_trace = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+text',
            text=token_labels,
            textposition="top center",
            marker=dict(
                size=16,
                color=colors[idx % len(colors)],
                line=dict(width=1, color='black')
            ),
            name=graph_info['name'],
            hoverinfo='text'
        )
        node_traces.append(node_trace)
        
        edge_x = []
        edge_y = []
        edge_index = graph_info['edge_index']
        for j in range(edge_index.shape[1]):
            src = edge_index[0, j]
            dst = edge_index[1, j]
            global_src = src + start
            global_dst = dst + start
            x0, y0 = all_embeddings_2d[global_src]
            x1, y1 = all_embeddings_2d[global_dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color=colors[idx % len(colors)]),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    fig = go.Figure(data=edge_traces + node_traces,
                    layout=go.Layout(
                        title=dict(text="2D PCA Projection of All Network Node Embeddings", x=0.5),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    st.subheader("Combined 2D Projection")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enable the checkbox above to display network visualizations.")
