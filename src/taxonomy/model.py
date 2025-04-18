
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GraphAutoencoder(nn.Module):
    def __init__(self, num_tokens, embed_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        # Embedding layer: maps token indices to vectors.
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        # Two GCN layers to produce latent node embeddings.
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.convfinal = GCNConv(hidden_dim, latent_dim)
        # Node decoder: a linear layer to predict the original token for each node.
        self.node_decoder = nn.Linear(latent_dim, num_tokens)

    def encode(self, data):
        # data.x: token indices for nodes.
        x = self.embedding(data.x)  # Shape: [total_nodes, embed_dim]
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = F.relu(self.conv4(x, data.edge_index))
        z = self.convfinal(x, data.edge_index)  # Shape: [total_nodes, latent_dim]
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
        return z, graph_emb, pred_adj, mask , node_logits


bert_model_name = 'bert-base-uncased'

class GraphBertAutoencoder(nn.Module):
    def __init__(self, num_tokens, embed_dim, hidden_dim, latent_dim):
        super(GraphBertAutoencoder, self).__init__()
        # Embedding layer: maps token indices to vectors.
        #self.embedding = nn.Embedding(num_tokens, embed_dim)

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Two GCN layers to produce latent node embeddings.
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.convfinal = GCNConv(hidden_dim, latent_dim)
        
        # Node decoder: a linear layer to predict the original token for each node.
        self.node_decoder = nn.Linear(latent_dim, embed_dim)
        self.norm_layer = nn.LayerNorm(embed_dim)
    def encode(self, data):
        self.bert.eval()
        # data.x: token indices for nodes.
        bert_embedding = x = self.bert(input_ids=data.x, attention_mask=data.attention_mask).pooler_output
        x = bert_embedding = self.norm_layer(x)
        #print(x.shape)
        #input()
        # Shape: [total_nodes, embed_dim]
        
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = F.relu(self.conv4(x, data.edge_index))
        #x = F.relu(self.conv2(x, data.edge_index))
        z = self.convfinal(x, data.edge_index)  # Shape: [total_nodes, latent_dim]
        return z, bert_embedding

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
        z,bert_encoding = self.encode(data)
        graph_emb = global_mean_pool(z, data.batch)
        pred_adj, mask = self.decode_adj(z, data.batch)
        node_logits = self.decode_node(z)
        #print(node_logits.shape,bert_encoding.shape,pred_adj.shape,mask.shape)
        return z, graph_emb, pred_adj, mask, node_logits, bert_encoding
