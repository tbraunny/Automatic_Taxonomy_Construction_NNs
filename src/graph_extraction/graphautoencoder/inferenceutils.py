import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


import os,sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



from src.graph_extraction.graphautoencoder.processing import parse_networks,LAYER_MAPPING,REVERSE_MAPPING,create_chain_graph



def load_dataset(filepath='./networks.json',passed_in_json=None,selected_model='fixed'):
    if passed_in_json == None:
        sample_json = open('./networks.json','r').read()
    else:
        sample_json = passed_in_json
    parsed_networks,string_parsed_networks = parse_networks(sample_json)
    #print(parsed_networks[2])
    if not parsed_networks:
        print("No valid networks found in the JSON.")
    else:
        print(f"Parsed {len(parsed_networks)} networks successfully.")
    if selected_model == 'fixed':
        dataset = [create_chain_graph(net_name, tokens,orders=orders, ogstrings=originalStrings) for net_name, tokens, orders, originalStrings in parsed_networks]
    else:
        dataset = [create_chain_graph(net_name, tokens,orders=orders, text=True, ogstrings=originalStrings) for net_name, tokens, orders, originalStrings in string_parsed_networks]
    return parsed_networks,string_parsed_networks,dataset

def loadModel(selected_model,device='cuda'):
    if selected_model  == 'fixed':
        path = 'src/graph_extraction/graphautoencoder/test.pt'
    else:
        path = 'src/graph_extraction/graphautoencoder/testbert.pt'
    model = torch.load(path)#GraphAutoencoder(num_tokens=num_tokens, embed_dim=8, hidden_dim=16, latent_dim=8)
    model = model.to(device)
    model.eval()
    return model

def get_embedding_dataset(model,selected_model, dataset,device='cuda'):
    graph_boundaries = []
    for index, data_obj in enumerate(dataset):
        #if not data_obj['name'] in selected:
        #    continue

        #print(data_obj.name)
        data_obj = data_obj.to(device)
        with torch.no_grad():
            if selected_model == 'fixed':
                z = model.encode(data_obj)  # [num_nodes, latent_dim]
            else:
                z,bert_embedding = model.encode(data_obj)  # [num_nodes, latent_dim]
        print(f"computed {data_obj['name']}")
        num_nodes = z.size(0)
        #all_embeddings_list.append(z.cpu())
        graph_boundaries.append({
            'name': data_obj.name,
            'start': index,
            'end': index + num_nodes,
            'edge_index': data_obj.edge_index.cpu().numpy(),
            'embedding': z
        })
        #current_index += num_nodes
        #indices.append(index)
    return graph_boundaries
