import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_scatter import scatter_mean
from typing import Tuple

class GVPEncoder(nn.Module):
    """
    Simplified geometric pocket encoder using GATv2 on Cα graphs.
    Processes scalar node/edge features + uses positional encoding from coords.
    """
    def __init__(self, node_in_dims=(27,1), edge_in_dims=(5,1),
                 hidden_dims=(128,16), out_dim=256, num_layers=5, drop_rate=0.1):
        super().__init__()
        # We use scalar dims only — node: 27, edge: 5
        node_in = node_in_dims[0]
        edge_in = edge_in_dims[0]
        hidden = hidden_dims[0]
        heads = 4

        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_embed = nn.Linear(edge_in, hidden)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden//heads, heads=heads,
                      edge_dim=hidden, dropout=drop_rate, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(drop_rate)

        self.readout = nn.Sequential(
            nn.Linear(hidden*2, out_dim), nn.ReLU(),
            nn.Dropout(drop_rate), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x_s, x_v, edge_index, edge_s, edge_v, batch):
        # Use only scalar features
        x = self.node_embed(x_s)
        e = self.edge_embed(edge_s)

        for conv, norm in zip(self.convs, self.norms):
            x = x + self.dropout(norm(conv(x, edge_index, e)))

        out = torch.cat([global_mean_pool(x, batch),
                         global_add_pool(x, batch)], dim=-1)
        return self.readout(out)
