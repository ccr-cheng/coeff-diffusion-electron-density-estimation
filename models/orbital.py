import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter


class GaussianOrbital(nn.Module):
    def __init__(self, gauss_start, gauss_end, n_gauss):
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.n_gauss = n_gauss

        self.sigma_r: torch.Tensor
        self.sigma_2r: torch.Tensor
        sigma = torch.linspace(gauss_start, gauss_end, n_gauss)
        self.register_buffer('sigma_r', 1 / (sigma * np.sqrt(2 * np.pi)).pow(3))
        self.register_buffer('sigma_2r', 1 / sigma.pow(2))

    def forward(self, coeff, atom_coord, grid, batch):
        """
        Expand the gaussian orbital to the grid points
        :param coeff: coefficients of the gaussian orbital, shape (N, *, n_gauss)
        :param atom_coord: atom coordinates, shape (N, 3)
        :param grid: grid coordinates, shape (G, K, 3)
        :param batch: batch index, shape (N,)
        :return: density on the grid points, shape (G, *, K)
        """
        sample_vec = grid[batch] - atom_coord.unsqueeze(-2)  # (N, K, 3)
        r2 = sample_vec.pow(2).sum(dim=-1)  # (N, K)
        orbital = torch.exp(-r2.unsqueeze(-1) * self.sigma_2r) * self.sigma_r  # (N, K, n_gauss)
        if coeff.dim() == 3:
            orbital = orbital.unsqueeze(-3)  # (N, *, K, n_gauss)
        density = (orbital * coeff.unsqueeze(-2)).sum(dim=-1)
        density = scatter(density, batch, dim=0, reduce='sum')
        return density
