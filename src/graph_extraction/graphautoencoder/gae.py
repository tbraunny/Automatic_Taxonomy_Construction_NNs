import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from model import GraphAutoencoder,GraphBertAutoencoder


from transformers import AutoTokenizer


# =============================================================================
# 1. PARSING NETWORK DEFINITIONS INTO TOKEN SEQUENCES
# =============================================================================

# Define a lowercase mapping of known (base) layer names to tokens.


# New token mapping dictionary for layers (all keys are lowercase).
# Similar or synonymous terms are mapped to the same token.
LAYER_MAPPING = {
    # Standard Keras layers (and equivalents)
    "conv2d": 1,
    "maxpooling2d": 2,
    "dropout": 3,
    "flatten": 4,
    "dense": 5,
    "linear": 5,              # Treat linear as dense
    "lstm": 6,
    "embedding": 7,
    "batchnormalization": 8,
    "activation": 9,          # Generic activation fallback
    "lambda": 10,
    "cropping2d": 11,
    "conv1d": 12,
    "spatialdropout1d": 13,
    "repeatvector": 14,
    "upsampling3d": 15,
    "upsampling2d": 15,
    "upsampling1d": 15,       # Same as upsampling2d
    "reshape": 16,
    "zeropadding2d": 17,
    "zeropadding1d": 17,      # Map 1D version to same as 2D
    "averagepooling2d": 18,
    "averagepooling1d": 18,
    "averagepooling3d": 18,
    "maxpooling1d": 19,
    "maxpooling2d": 19,
    "maxpooling3d": 19,
    "globalaveragepooling1d": 20,
    "globalaveragepooling2d": 20,
    "globalaveragepooling3d": 20,
    "bidirectional": 21,
    "leakyrelu": 22,
    "timedistributed": 23,
    "maxout": 24,
    "merge": 25,
    "relu": 26,
    "softmax": 27,
    "conv3d": 28,
    "maxpooling3d": 29,
    "tanh": 30,
    "sigmoid": 31,
    "gru": 32,
    "zeropadding3d": 33,
    "input": 34,
    "elu": 35,
    "concatenate": 36,
    "concat": 36,
    "gaussiannoise": 37,
    "masking": 38,
    "attention": 39,          # Corrected spelling (also used for SSA)
    "ssa": 39,                # Map "ssa" (self-attention) to same as attention
    "permute": 40,
    "sum": 41,
    "softsign": 42,
    "softplus": 43,
    "spatialtransformer": 61,
    "constantlayer": 62,
    "capsule": 63,
    "layernormalization": 63,
    "graphcnn":64,
    "gn":65,
    "autoencoder":66,
    # Additional Keras/custom layers and synonyms:
    "activityregularization": 44,
    "locallyconnected1d": 45,
    "locallyconnected2d": 45,
    "locallyconnected3d": 45,
    "simplernn": 46,
    "conv1": 12,              # Map shorthand "conv1" to conv1d
    "bias": 47,
    "maxpool": 2,             # Map shorthand "maxpool" to maxpooling2d
    "batchnorm": 8,           # Map shorthand "bn" also to 8
    "bn": 8,
    "fc": 5,                  # Fully-connected is the same as dense
    "rnn": 46,                # Map generic "rnn" to simplernn
    "resnet50_1": 48,         # Custom model indicator for resnet50
    "inceptionv3_1": 49,      # Custom inception variant
    "xception_1": 50,         # Custom xception variant
    "lru": 51,                # Possibly a custom layer; assign new token
    "deconvolution2d": 52,    # Also known as transposed conv
    "brnn": 21,               # Bidirectional RNN (map to bidirectional)
    "groupnormalization": 53, # GroupNorm layer
    "seq2seq": 54,            # Sequence-to-sequence model layer
    "ccmmembership": 55,       # Custom layer (if used)
    "vgg16": 56,              # A network indicator (could be used as a token)
    "incept": 49,             # Shorthand for inception (same as inceptionv3_1)
    "normalizeimage": 57,     # Preprocessing layer for image normalization
    'globalpooling1d':58,
    'globalpooling2d':58,
    'globalpooling3d':58,
    "graphconv": 59,          # Graph convolution (if used in a model)
    "defuzzylayer": 60        # Custom layer for defuzzification
}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def encode_node_text(text, max_length=32):
    """
    Encodes a text string using BERT and returns the [CLS] embedding.
    """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    #with torch.no_grad():
    #    outputs = bert_model(**inputs)
    # Use the [CLS] token (first token) embedding as the representation.
    #cls_embedding = outputs.last_hidden_state[0, 0, :]
    #print(inputs)
    #input()
    return inputs

def match_layer_name(name_clean):
    """
    Given a cleaned (lowercase, stripped) layer name, first try an exact match.
    If that fails, try a partial match: if any key from LAYER_MAPPING is found as a substring,
    return its token. Returns the token (an integer) or None if no match is found.
    """
    # Exact match:
    if name_clean in LAYER_MAPPING:
        return LAYER_MAPPING[name_clean]
    # Partial matching:
    for key in LAYER_MAPPING:
        if key in name_clean:
            return LAYER_MAPPING[key]
    return None

def parse_networks(json_str, return_strings=False):
    """
    Parses a JSON string containing a list of network definitions.
    Each network must have a "name" and a "network" field (a list of layer objects).
    Each layer object must have:
      - "type": e.g. "Layer", "Activation", or "Custom"
      - "name": the layer name (which may include extra punctuation or suffixes, e.g. "leakyrelu_3")
      - "order": an integer specifying its order in the network.
    
    Returns a list of tuples: (network_name, token_list) for each valid network.
    A network is rejected if any layer cannot be mapped.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
        return []
    
    valid_networks = []
    string_networks = []
    count = 2000
    for net_obj in data:
        #if count == 0:
        #    break
        #count -= 1
        network_name = net_obj.get("name", "unknown")
        layers = net_obj.get("network", [])
        # Sort layers by "order"
        sorted_layers = sorted(layers, key=lambda x: x.get("order", 0))
        tokens = []
        strings = [] 
        valid = True
        
        for layer in sorted_layers:
            layer_type = layer.get("type", "").strip().lower()
            layer_name_raw = layer.get("name", "")
            # Clean up the layer name: remove brackets, quotes, and extra whitespace; then lowercase.
            name_clean = layer_name_raw.strip("[]'\" ").lower()
            token = match_layer_name(name_clean)
            strings.append(   encode_node_text (name_clean) )
            if token is None:
                # If this is an Activation layer, fallback to the generic "activation" token.
                if layer_type == "activation":
                    token = LAYER_MAPPING["activation"]
                    if 'not' in name_clean:
                        valid = False
                        #break
                    #print(layer)
                    #input()
                else:
                    print(f"Unknown layer '{layer_name_raw}' (cleaned: '{name_clean}') in network {network_name}. Rejecting network.")
                    valid = False
                    #print(layer)
                    #input()
                    #break
            tokens.append(token)
        if valid:
            valid_networks.append((network_name, tokens))
        string_networks.append((network_name, strings))
    return valid_networks, string_networks

# =============================================================================
# 2. CONVERT TOKEN SEQUENCE INTO A CHAIN GRAPH (PyTorch Geometric Data object)
# =============================================================================


def process_token_list(token_list, key):
    """
    Given a list of dictionaries (one per node) with a key (e.g. 'input_ids' or 'attention_mask'),
    convert each to a 1D tensor. If the tensor has an extra singleton dimension (shape [1, L]),
    squeeze it out so that the result is shape [L].
    Returns a tensor of shape [num_nodes, L].
    """
    tensors = []
    for tokens in token_list:
        t = torch.tensor(tokens[key], dtype=torch.long)
        # If t has shape [1, L], squeeze the first dimension.
        if t.dim() == 2 and t.size(0) == 1:
            t = t.squeeze(0)
        tensors.append(t)
    return torch.stack(tensors, dim=0)

def create_chain_graph(network_name, token_list,text=False):
    """
    Given a network name and a list of integer tokens (one per layer, in order),
    creates a chain graph: each node corresponds to a layer, and nodes are connected
    in sequence. Returns a PyTorch Geometric Data object.
    """

    num_nodes = len(token_list)
    #print(num_nodes)
    #input()
    # Build edge_index for a chain. We add edges in both directions to simulate an undirected graph.
    if num_nodes > 1:
        # Create edges from i to i+1 and i+1 to i for i in [0, num_nodes-2]
        src = []
        dst = []
        for i in range(num_nodes - 1):
            src.append(i)
            dst.append(i + 1)
            src.append(i + 1)
            dst.append(i)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # If only one node, no edges.
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Optionally, you can store the network name as an attribute.
    if not text:

        # Node features: store tokens as a 1D tensor.
        x = torch.tensor(token_list, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)


    else:
        try:
            #x = torch.stack([torch.tensor(tokens['input_ids'], dtype=torch.long)
            #                 for tokens in token_list], dim=0)
            #attention_mask = torch.stack([torch.tensor(tokens['attention_mask'], dtype=torch.long)
            x = process_token_list(token_list, 'input_ids')
            attention_mask = process_token_list(token_list, 'attention_mask')
                #data = Data(
            #    x=torch.stack([torch.tensor(tokens['input_ids'], dtype=torch.long) for tokens in token_list], dim=0),
            #    attention_mask=torch.stack([torch.tensor(tokens['attention_mask'], dtype=torch.long) for tokens in token_list], dim=0),
            #    edge_index=edge_index
            #)
        except Exception as e:
            raise ValueError(f"Error processing token_list with text data: {e}")
        data = Data(x=x, edge_index=edge_index, attention_mask=attention_mask)
        #print(data)
        #print(num_nodes)
        #input()
    data.name = network_name
    return data

# =============================================================================
# 3. DEFINE THE GRAPH AUTOENCODER MODEL
# =============================================================================


'''class GraphAutoencoder(nn.Module):
    def __init__(self, num_tokens, embed_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        # Embedding layer: maps token indices to vectors.
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        # Two GCN layers to produce latent node embeddings.
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        # Node decoder: a linear layer to predict the original token for each node.
        self.node_decoder = nn.Linear(latent_dim, num_tokens)

    def encode(self, data):
        # data.x: token indices for nodes.
        x = self.embedding(data.x)  # Shape: [total_nodes, embed_dim]
        x = F.relu(self.conv1(x, data.edge_index))
        z = self.conv2(x, data.edge_index)  # Shape: [total_nodes, latent_dim]
        return z

    def decode_adj(self, z, batch):
        """
        Given latent node embeddings z and a batch vector, convert to a dense batch
        and compute the inner-product reconstruction of the adjacency matrix.
        Returns:
          pred_adj: Tensor of shape [B, max_nodes, max_nodes]
          mask: Boolean mask tensor for valid nodes.
        """
        z_dense, mask = to_dense_batch(z, batch=batch)
        pred_adj = torch.sigmoid(torch.bmm(z_dense, z_dense.transpose(1, 2)))
        return pred_adj, mask

    def decode_node(self, z):
        """
        Decodes each node’s latent embedding to predict the original token.
        Returns logits of shape [total_nodes, num_tokens].
        """
        logits = self.node_decoder(z)
        return logits

    def forward(self, data):
        """
        Forward pass that returns:
          - z: latent node embeddings,
          - graph_emb: graph-level embedding via global mean pooling,
          - pred_adj: reconstructed dense adjacency matrix,
          - mask: Boolean mask for valid nodes (in the dense batch),
          - node_logits: logits for reconstructing node features.
        """
        z = self.encode(data)
        graph_emb = global_mean_pool(z, data.batch)
        pred_adj, mask = self.decode_adj(z, data.batch)
        node_logits = self.decode_node(z)
        return z, graph_emb, pred_adj, mask, node_logits
'''
# =============================================================================
# 4. MAIN SCRIPT: PARSE, CONVERT TO GRAPHS, AND TRAIN THE AUTOENCODER
# =============================================================================
if __name__ == "__main__":
    # --- Sample JSON string (shortened example) ---
    sample_json = open('networks.json','r').read() 
    '''
    [
        {
            "name": "https://w3id.org/nno/data#zoogzog/deeplearning_model_2",
            "network": [
                {"type": "Layer", "name": "Conv2D", "order": 1},
                {"type": "Layer", "name": "Conv2D", "order": 2},
                {"type": "Layer", "name": "MaxPooling2D", "order": 3},
                {"type": "Layer", "name": "Dropout", "order": 4},
                {"type": "Layer", "name": "Flatten", "order": 5},
                {"type": "Layer", "name": "Dense", "order": 6},
                {"type": "Layer", "name": "Dropout", "order": 7},
                {"type": "Layer", "name": "Dense", "order": 8}
            ]
        },
        {
            "name": "https://w3id.org/nno/data#zpy009/short-video-classification_model_1",
            "network": [
                {"type": "Custom", "name": "bidirectional_1", "order": 1},
                {"type": "Custom", "name": "bidirectional_2", "order": 2},
                {"type": "Layer", "name": "Dense", "order": 3},
                {"type": "Layer", "name": "Dropout", "order": 4},
                {"type": "Layer", "name": "Dense", "order": 5},
                {"type": "Activation", "name": "['softmax']", "order": 6}
            ]
        },
        {
            "name": "https://w3id.org/nno/data#unknown_model",
            "network": [
                {"type": "Layer", "name": "Conv2D", "order": 1},
                {"type": "Custom", "name": "unknown_custom_layer", "order": 2},
                {"type": "Layer", "name": "Dense", "order": 3}
            ]
        },
        {
            "name": "https://w3id.org/nno/data#model_with_leakyrelu",
            "network": [
                {"type": "Layer", "name": "Conv2D", "order": 1},
                {"type": "Custom", "name": "leakyrelu_3", "order": 2},
                {"type": "Layer", "name": "Dense", "order": 3}
            ]
        }
    ]
    '''

    # --- Parse the JSON into token sequences ---
    parsed_networks, parsed_string_networks = parse_networks(sample_json)
    print("Parsed networks (only valid ones are kept):")
    for net_name, tokens in parsed_networks:
        print(f"  {net_name} -> Tokens: {tokens}")
    
    # --- Convert each token sequence into a chain graph ---
    dataset = []
    for net_name, tokens in parsed_networks:
        data_obj = create_chain_graph(net_name, tokens)

        dataset.append(data_obj)
  
    text_dataset = []
    for net_name, strings in parsed_string_networks:
        #print(strings)
        #print(len([tokens['input_ids'] for tokens in strings]) ) 
        #print(
        data_obj = create_chain_graph(net_name, strings, text=True)
        #print('strings',data_obj)
        #input()
        text_dataset.append(data_obj)


    
    # --- Instantiate the Graph Autoencoder model ---
    # Use max(LAYER_MAPPING.values()) + 1 as the number of tokens for embedding.

    training_text=True
    if not training_text:
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # --- Create a DataLoader ---
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) 
        num_tokens = max(LAYER_MAPPING.values()) + 1
        model = GraphAutoencoder(num_tokens=num_tokens, embed_dim=768, hidden_dim=256, latent_dim=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        # Use ReduceLROnPlateau scheduler to reduce LR when validation loss plateaus.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        scheduler3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)
        num_epochs = 10000
        print("\nStarting training with validation and LR scheduler...\n")
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            total_train_node_loss = 0.0
            total_train_adj_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                z, graph_emb, pred_adj, mask, node_logits = model(batch)
                #z, graph_emb, pred_adj, mask = model(batch)
                ground_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                #loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                #loss_adj = F.mse_loss(pred_adj, ground_adj)
                loss_node = F.cross_entropy(node_logits, batch.x)
                loss = loss_adj + loss_node

                #loss = loss_adj
                loss.backward()
                optimizer.step()
                total_train_adj_loss += loss_adj.item() * batch.num_graphs
                total_train_node_loss += loss_node.item() * batch.num_graphs
                total_train_loss += loss.item() * batch.num_graphs
            avg_train_loss = total_train_loss / len(train_dataset)
            avg_node_loss = total_train_node_loss / len(train_dataset) 
            avg_adj_loss = total_train_adj_loss / len(train_dataset) 
            # Validation pass.
            model.eval()
            total_val_loss = 0.0
            total_val_node_loss = 0.0
            total_val_adj_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    z, graph_emb, pred_adj, mask, node_logits = model(batch)
                    #z, graph_emb, pred_adj, mask = model(batch)
                    ground_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                    mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                    #loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                    loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                    loss_node = F.cross_entropy(node_logits, batch.x)
                    loss = loss_adj + loss_node
                    total_val_node_loss += loss_node.item() * batch.num_graphs
                    total_val_adj_loss += loss_adj.item() * batch.num_graphs
                    total_val_loss += loss.item() * batch.num_graphs
            avg_val_loss = total_val_loss / len(val_dataset)
            avg_val_adj_loss = total_val_adj_loss / len(val_dataset)
            avg_val_node_loss = total_val_node_loss / len(val_dataset)
            scheduler.step(avg_val_loss)  # Update LR if validation loss plateaus
            #scheduler1.step()
            #scheduler2.step()
            scheduler3.step(epoch/10)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss, Node Loss, Adj Loss: {avg_train_loss:.4f},{avg_node_loss:.4f},{avg_adj_loss:.4f} - Val Loss,node,adj: {avg_val_loss:.4f},{avg_val_node_loss:.4f},{avg_val_adj_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.5f}")
        
        # --- Evaluation: Print sample outputs from validation set ---
        torch.save(model,'test.pt')
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader)).to(device)
            z, graph_emb, pred_adj, mask, node_logits = model(sample_batch)
            ground_adj = to_dense_adj(sample_batch.edge_index, batch=sample_batch.batch)
            pred_first = pred_adj[0]
            ground_first = ground_adj[0]
            valid_nodes = mask[0].bool()
            print("\nReconstructed adjacency matrix for first graph (valid entries):")
            print(pred_first[valid_nodes][:, valid_nodes])
            print("\nGround-truth adjacency matrix for first graph (valid entries):")
            print(ground_first[valid_nodes][:, valid_nodes])
            print("\nGraph-level embedding for first graph:")
            print(graph_emb[0])
            
            # Additionally, show node-level reconstruction accuracy:
            # Get the predicted token class (via argmax) for each node.
            node_pred = node_logits.argmax(dim=1)
            print("\nOriginal node tokens for first graph:")
            # We need to extract the node tokens corresponding to the first graph.
            # Using to_dense_batch to obtain padded node tokens.
            x_dense, node_mask = to_dense_batch(sample_batch.x, batch=sample_batch.batch)
            print(x_dense[0][node_mask[0]])
            print("\nReconstructed node tokens for first graph:")
            # Split node_logits into a dense batch.
            node_logits_dense, _ = to_dense_batch(node_logits, batch=sample_batch.batch)
            node_pred_dense = node_logits_dense.argmax(dim=2)
            print(node_pred_dense[0][node_mask[0]]) 
    else:
        num_tokens = max(LAYER_MAPPING.values()) + 1
        model = GraphBertAutoencoder(num_tokens=num_tokens, embed_dim=768, hidden_dim=256, latent_dim=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,amsgrad=True)
        
        # Use ReduceLROnPlateau scheduler to reduce LR when validation loss plateaus.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        scheduler3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)
        
        num_epochs = 300
        print("\nStarting training with validation and LR scheduler...\n")
        print(len(text_dataset))
        train_size = int(0.7 * len(text_dataset))
        val_size = len(text_dataset) - train_size
        train_dataset, val_dataset = random_split(text_dataset, [train_size, val_size])

        # --- Create a DataLoader ---
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False) 
        print(len(train_loader))
        for i in train_loader:
            print(i)
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            total_train_node_loss = 0.0
            total_train_adj_loss = 0.0
            for batch in train_loader:
                #print(batch)
                batch = batch.to(device)
                #print(batch.edge_index[0])
                #input()
                optimizer.zero_grad()
                z, graph_emb, pred_adj, mask, node_logits,bert_encoding = model(batch)
                #z, graph_emb, pred_adj, mask = model(batch)
                ground_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                #print(pred_adj.shape,ground_adj.shape)
                #loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                #print(loss_adj)
                #loss_adj += F.binary_cross_entropy(pred_adj, ground_adj)
                #loss_adj = F.binary_cross_entropy(pred_adj, ground_adj)
                #loss_adj = F.cross_entropy(pred_adj, ground_adj)
                #loss_adj = F.mse_loss(pred_adj, ground_adj)
                #loss_node = F.cross_entropy(node_logits, batch.x)
                #print(batch.batch.shape)
                #print(mask2d.shape)
                loss_node = F.mse_loss(node_logits, bert_encoding)
                #print(pred_adj)
                #print(ground_adj)
                #input()
                loss = loss_adj #+ loss_node
                #loss = loss_adj
                loss.backward()
                optimizer.step()
                total_train_adj_loss += loss_adj.item() * batch.num_graphs
                total_train_node_loss += loss_node.item() * batch.num_graphs
                total_train_loss += loss.item() * batch.num_graphs
            avg_train_loss = total_train_loss / len(train_dataset)
            avg_node_loss = total_train_node_loss / len(train_dataset) 
            avg_adj_loss = total_train_adj_loss / len(train_dataset) 
            # Validation pass.
            model.eval()
            total_val_loss = 0.0
            total_val_node_loss = 0.0
            total_val_adj_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    z, graph_emb, pred_adj, mask, node_logits, bert_encoding = model(batch)
                    #z, graph_emb, pred_adj, mask = model(batch)
                    ground_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
                    mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                    #loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])

                    loss_adj = F.binary_cross_entropy(pred_adj[mask2d], ground_adj[mask2d])
                    #loss_adj = F.binary_cross_entropy(pred_adj, ground_adj)
                    #loss_node = F.cross_entropy(node_logits, batch.x)
                    loss_node = F.mse_loss(node_logits, bert_encoding)
                    loss = loss_adj + loss_node
                    total_val_node_loss += loss_node.item() * batch.num_graphs
                    total_val_adj_loss += loss_adj.item() * batch.num_graphs
                    total_val_loss += loss.item() * batch.num_graphs
            avg_val_loss = total_val_loss / len(val_dataset)
            avg_val_adj_loss = total_val_adj_loss / len(val_dataset)
            avg_val_node_loss = total_val_node_loss / len(val_dataset)
            scheduler.step(avg_val_loss)  # Update LR if validation loss plateaus
            #scheduler1.step()
            #scheduler2.step()
            scheduler3.step(epoch/10)
        
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss, Node Loss, Adj Loss: {avg_train_loss:.4f},{avg_node_loss:.4f},{avg_adj_loss:.4f} - Val Loss,node,adj: {avg_val_loss:.4f},{avg_val_node_loss:.4f},{avg_val_adj_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.5f}")
        # --- Evaluation: Print sample outputs from validation set ---
        torch.save(model,'testbert.pt')
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader)).to(device)
            z, graph_emb, pred_adj, mask, node_logits,bert_encoding = model(sample_batch)
            ground_adj = to_dense_adj(sample_batch.edge_index, batch=sample_batch.batch)
            pred_first = pred_adj[0]
            ground_first = ground_adj[0]
            valid_nodes = mask[0].bool()
            print("\nReconstructed adjacency matrix for first graph (valid entries):")
            print(pred_first[valid_nodes][:, valid_nodes])
            print("\nGround-truth adjacency matrix for first graph (valid entries):")
            print(ground_first[valid_nodes][:, valid_nodes])
            print("\nGraph-level embedding for first graph:")
            print(graph_emb[0])
            
            # Additionally, show node-level reconstruction accuracy:
            # Get the predicted token class (via argmax) for each node.
            node_pred = node_logits.argmax(dim=1)
            print("\nOriginal node tokens for first graph:")
            # We need to extract the node tokens corresponding to the first graph.
            # Using to_dense_batch to obtain padded node tokens.
            x_dense, node_mask = to_dense_batch(sample_batch.x, batch=sample_batch.batch)
            print(x_dense[0][node_mask[0]])
            #print("\nReconstructed node tokens for first graph:")
            # Split node_logits into a dense batch.
            #node_logits_dense, _ = to_dense_batch(node_logits, batch=sample_batch.batch)
            #node_pred_dense = node_logits_dense.argmax(dim=2)
            #print(node_pred_dense[0][node_mask[0]]) 
