import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

from ._base import register_encoder
from .utils import MLP, get_dist_embed, get_timestep_embedding


class GCNLayer(nn.Module):
    def __init__(self, node_feat_size, node_attr_size, edge_attr_size, hidden_size=1024):
        super(GCNLayer, self).__init__()
        self.node_feat_size = node_feat_size
        self.node_attr_size = node_attr_size
        self.edge_attr_size = edge_attr_size
        self.hidden_size = hidden_size

        self.src_fc = MLP(
            node_feat_size + node_attr_size + edge_attr_size,
            [hidden_size, hidden_size * 2, hidden_size],
            node_feat_size
        )
        self.dst_fc = MLP(node_attr_size, [hidden_size, hidden_size * 2, hidden_size], node_feat_size)
        self.feat_fc = MLP(node_feat_size, [hidden_size, hidden_size * 2, hidden_size], node_feat_size)
        self.fc = nn.Linear(node_feat_size * 3, node_feat_size)

    def forward(self, edge_index, node_feat, node_attr, edge_attr):
        src, dst = edge_index
        dst_attr = self.dst_fc(node_attr)
        dst_feat = self.feat_fc(node_feat)
        src_feat = self.src_fc(torch.cat([
            node_feat[src], node_attr[src], edge_attr
        ], dim=-1))
        src_feat = scatter(src_feat, dst, dim=0, reduce='sum')
        out = self.fc(torch.cat([dst_attr, dst_feat, src_feat], dim=-1))
        return out


@register_encoder('gcn')
class GCNEncoder(nn.Module):
    def __init__(self, n_atom_type, n_gauss, atom_embed_size=256, radial_embed_size=256,
                 hidden_size=1024, num_gcn_layer=1, activation=nn.GELU(), cutoff=5., time_embed_size=0):
        super(GCNEncoder, self).__init__()
        self.n_atom_type = n_atom_type
        self.n_gauss = n_gauss
        self.atom_embed_size = atom_embed_size
        self.radial_embed_size = radial_embed_size
        self.hidden_size = hidden_size
        self.num_gcn_layer = num_gcn_layer
        self.activation = activation
        self.cutoff = cutoff
        self.time_embed_size = time_embed_size

        self.atom_embedding = nn.Embedding(n_atom_type, atom_embed_size)
        self.node_in_size = atom_embed_size + time_embed_size
        self.gcns = nn.ModuleList([
            GCNLayer(n_gauss, self.node_in_size, radial_embed_size, hidden_size)
            for _ in range(num_gcn_layer)
        ])

    def forward(self, atom_types, atom_coord, batch, node_feat=None, timestep=None):
        if node_feat is None:
            node_feat = torch.zeros(atom_types.size(0), self.n_gauss, device=atom_types.device)
        edge_index = radius_graph(atom_coord, self.cutoff, batch, loop=False)
        src, dst = edge_index

        node_attr = self.atom_embedding(atom_types)
        if timestep is not None:
            time_embed = get_timestep_embedding(timestep, self.time_embed_size)
            node_attr = torch.cat([node_attr, time_embed], dim=-1)
        edge_vec = atom_coord[src] - atom_coord[dst]
        edge_attr = get_dist_embed(edge_vec.norm(dim=-1), 0, self.cutoff, self.radial_embed_size)

        for i, gcn in enumerate(self.gcns):
            node_feat = gcn(edge_index, node_feat, node_attr, edge_attr)
            if i != len(self.gcns) - 1:
                node_feat = self.activation(node_feat)
        return node_feat
