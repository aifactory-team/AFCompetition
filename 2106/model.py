import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv
from dgl.nn.pytorch.glob import AvgPooling
from torch.nn import ModuleList


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super(MLPReadout, self).__init__()
        FC_layers = []
        # Add hidden FC layers
        for l in range(L):
            FC_layers += [
                nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True),
                nn.ReLU(),
            ]

        # Add output FC layer
        FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.Sequential(*FC_layers)

    def forward(self, x):
        x = self.FC_layers(x)
        return x


class GNN_model(torch.nn.Module):
    def __init__(self, graph_encoder_params):
        super(GNN_model, self).__init__()
        # graph
        self.graph_encoder_n_layer = graph_encoder_params["n_layer"]
        self.graph_encoder_dim = graph_encoder_params["hidden_dim"]
        self.num_atom_type = graph_encoder_params["num_atom_type"]
        self.embedding_h = nn.Embedding(self.num_atom_type, self.graph_encoder_dim)
        self.gnn_layers = ModuleList([])

        for _ in range(self.graph_encoder_n_layer):
            self.gnn_layers.append(
                GraphConv(
                    in_feats=self.graph_encoder_dim,
                    out_feats=self.graph_encoder_dim,
                    norm="both",
                )
            )
        self.pooling_layer = AvgPooling()
        self.MLP_layer = MLPReadout(self.graph_encoder_dim, 1)

    def forward(self, graph):
        h = graph.ndata["f"]
        h = self.embedding_h(h)  # [2344] -> [2344, 64]

        for layer in self.gnn_layers:
            h = layer(graph, h)  # [2344, 64] -> [2344, 64]

        graph_embedding = self.pooling_layer(graph, h)  # [2344, 64] -> [128, 64]
        outputs = self.MLP_layer(graph_embedding)

        return outputs
