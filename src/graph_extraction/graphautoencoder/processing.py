import torch
import json

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from transformers import AutoTokenizer
# =============================================================================
# 1. PARSING NETWORK DEFINITIONS INTO TOKEN SEQUENCES
# =============================================================================

# Define a lowercase mapping of known (base) layer names to tokens.
def reverse_dictionary(original_dict):
    """Reverses the keys and values of a dictionary. 
    If multiple keys have the same value, only the last key encountered will be retained."""
    return {value: key for key, value in original_dict.items()}

# New token mapping dictionary for layers (all keys are lowercase).
# Similar or synonymous terms are mapped to the same token.
LAYER_MAPPING = {
    # Standard Keras layers (and equivalents)
    "conv2d": 1,
    "maxpooling2d": 2,
    "dropout": 3,
    "flatten": 4,
    'fullyconnectedlayer':5,
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
    'clonelayer': 36, # TODO -- clone layer is not concat
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

REVERSE_MAPPING = reverse_dictionary(LAYER_MAPPING)

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

def parse_networks(json_str):
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
    if type(data) == dict:
        data = [data]
    for net_obj in data:
        legacy = False # fairnets use sequential order
        tokens = []
        strings = [] 
        originalStrings = []
        orders = []
        valid = True
        #if count == 0:
        #    break
        #count -= 1
        network_name = net_obj.get("name", "unknown")
        layers = net_obj.get("network", [])
        if layers != []:
            # Sort layers by "order"
            layers = sorted_layers = sorted(layers, key=lambda x: x.get("order", 0))
            legacy = True
        else:
            layers = net_obj.get("nodes",[])
            #inputs = {index : layer['input']  for index,layer in enumerate(layers)}
            targets = { layer['name'] : index for index,layer in enumerate(layers)}
            for index, layer in enumerate(layers):
                for target in layer['target']:
                    if target in targets:
                        orders.append((index,targets[target]))
        
        for index,layer in enumerate(layers):
            if legacy and len(layers) > 1 and index < len(layers) - 1:
                orders.append((index,index+1))

            layer_type = layer.get("type", "").strip().lower()
            layer_name_raw = layer.get("name", "")
            if not legacy:
                layer_name_raw = layer.get("op", "") # TODO: need to add types as a part of exported data 
            # Clean up the layer name: remove brackets, quotes, and extra whitespace; then lowercase.
            name_clean = layer_name_raw.strip("[]'\" ").lower()
            token = match_layer_name(name_clean)
            originalStrings.append(name_clean)
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
            valid_networks.append((network_name, tokens, orders, originalStrings)) # TODO will merge below
            
        string_networks.append((network_name, strings, orders, originalStrings))
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

def create_chain_graph(network_name, token_list, orders=[], text=False, ogstrings=[]):
    """
    Given a network name and a list of integer tokens (one per layer, in order),
    creates a chain graph: each node corresponds to a layer, and nodes are connected
    in sequence. Returns a PyTorch Geometric Data object.
    """

    num_nodes = len(token_list)
    #print(num_nodes)
    #input()
    # Build edge_index for a chain. We add edges in both directions to simulate an undirected graph.
    
    if num_nodes > 1 and len(orders) == 0:
        # Create edges from i to i+1 and i+1 to i for i in [0, num_nodes-2]
        src = []
        dst = []
        for i in range(num_nodes - 1):
            src.append(i)
            dst.append(i + 1)
            src.append(i + 1)
            dst.append(i)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    elif len(orders) > 0:
        src = [order[0] for order in orders]
        dst = [order[1] for order in orders]
        edge_index = torch.tensor([src,dst], dtype=torch.long)
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
            #print(token_list)
            x = process_token_list(token_list, 'input_ids')
            attention_mask = process_token_list(token_list, 'attention_mask')
                #data = Data(
            #    x=torch.stack([torch.tensor(tokens['input_ids'], dtype=torch.long) for tokens in token_list], dim=0),
            #    attention_mask=torch.stack([torch.tensor(tokens['attention_mask'], dtype=torch.long) for tokens in token_list], dim=0),
            #    edge_index=edge_index
            #)
        except Exception as e:
            raise ValueError(f"Error processing token_list with text data: {e}")
        data = Data(x=x, edge_index=edge_index, attention_mask=attention_mask,ogstring=ogstrings)
        #print(data)
        #print(num_nodes)
    data.name = network_name
    return data
