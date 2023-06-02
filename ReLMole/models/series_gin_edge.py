import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter

dtype = torch.float32


class SerGINE(nn.Module):
    def __init__(self, num_atom_layers=3, num_fg_layers=2, latent_dim=128,
                 atom_dim=101, fg_dim=73, bond_dim=11, fg_edge_dim=101,
                 atom2fg_reduce='mean', pool='mean', dropout=0, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atom_layers = num_atom_layers
        self.num_fg_layers = num_fg_layers
        self.atom2fg_reduce = atom2fg_reduce

        # embedding
        self.atom_embedding = nn.Linear(atom_dim, latent_dim)
        self.fg_embedding = nn.Linear(fg_dim, latent_dim)
        self.bond_embedding = nn.ModuleList(
            [nn.Linear(bond_dim, latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_edge_embedding = nn.ModuleList(
            [nn.Linear(fg_edge_dim, latent_dim) for _ in range(num_fg_layers)]
        )

        # gnn
        self.atom_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_atom_layers)]
        )
        self.atom_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_atom_layers)]
        )
        self.fg_gin = nn.ModuleList(
            [GINEConv(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim*2), nn.BatchNorm1d(latent_dim*2), nn.ReLU(), nn.Linear(latent_dim*2, latent_dim)
                )
            ) for _ in range(num_fg_layers)]
        )
        self.fg_bn = nn.ModuleList(
            [nn.BatchNorm1d(latent_dim) for _ in range(num_fg_layers)]
        )
        self.atom2fg_lin = nn.Linear(latent_dim, latent_dim)
        # pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling!")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        atom_x, atom_edge_index, atom_edge_attr, atom_batch = data.x, data.edge_index, data.edge_attr, data.batch
        fg_x, fg_edge_index, fg_edge_attr, fg_batch = data.fg_x, data.fg_edge_index, data.fg_edge_attr, data.fg_x_batch
        atom_idx, fg_idx = data.atom2fg_index

        # one-hot to vec
        atom_x = self.atom_embedding(atom_x)
        fg_x = self.fg_embedding(fg_x)

        # atom-level gnn
        for i in range(self.num_atom_layers):
            atom_x = self.atom_gin[i](atom_x, atom_edge_index, self.bond_embedding[i](atom_edge_attr))
            atom_x = self.atom_bn[i](atom_x)
            if i != self.num_atom_layers-1:
                atom_x = self.relu(atom_x)
            atom_x = self.dropout(atom_x)

        # atom-level to FG-level
        atom2fg_x = scatter(atom_x[atom_idx], index=fg_idx, dim=0, dim_size=fg_x.size(0), reduce=self.atom2fg_reduce)
        atom2fg_x = self.atom2fg_lin(atom2fg_x)
        fg_x = fg_x + atom2fg_x

        # fg-level gnn
        for i in range(self.num_fg_layers):
            fg_x = self.fg_gin[i](fg_x, fg_edge_index, self.fg_edge_embedding[i](fg_edge_attr))
            fg_x = self.fg_bn[i](fg_x)
            if i != self.num_fg_layers-1:
                fg_x = self.relu(fg_x)
            fg_x = self.dropout(fg_x)

        fg_graph = self.pool(fg_x, fg_batch)

        return fg_graph
