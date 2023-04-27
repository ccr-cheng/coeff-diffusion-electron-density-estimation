import math

import torch
import torch.nn as nn


def get_timestep_embedding(timestep, embed_size):
    half_dim = embed_size // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timestep.device) * -emb)
    emb = timestep[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_size % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_dist_embed(x, start, end, embed_size):
    values = torch.linspace(start, end, embed_size, dtype=x.dtype, device=x.device)
    step = values[1] - values[0]
    diff = (x[..., None] - values) / step
    return diff.pow(2).neg().exp().div(1.12)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.GELU()):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.activation = activation

        dims = [in_dim] + hidden_dims + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x
