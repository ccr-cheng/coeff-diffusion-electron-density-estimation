from ._base import get_encoder
from .gnn import GCNEncoder
from .diffnet import DiffNet
from .orbital import GaussianOrbital


def get_model(model_cfg, device):
    m_cfg = model_cfg.copy()

    # orbital
    orbital_cfg = m_cfg.pop('orbital')
    orbital_cfg['n_gauss'] = m_cfg['n_gauss']
    _orbital = GaussianOrbital(**orbital_cfg).to(device)

    # diffnet
    diffnet_cfg = m_cfg.pop('diffusion')
    _diffnet = DiffNet(m_cfg, **diffnet_cfg).to(device)

    # predictor
    predictor = get_encoder(m_cfg).to(device)

    return predictor, _diffnet, _orbital
