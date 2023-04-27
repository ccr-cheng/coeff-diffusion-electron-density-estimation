import math

import torch

_BETA_DICT = {}


def register_beta(name):
    def decorator(cls):
        _BETA_DICT[name] = cls
        return cls

    return decorator


def get_beta(cfg):
    b_cfg = cfg.copy()
    b_type = b_cfg.pop('type')
    return _BETA_DICT[b_type](**b_cfg)


@register_beta('linear')
def get_linear_beta(beta_step, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, beta_step, dtype=torch.float)


@register_beta('cosine')
def get_cosine_beta(beta_step, s=0.008, max_beta=0.999):
    steps = beta_step + 1
    t = torch.linspace(0, beta_step, steps, dtype=torch.float) / beta_step
    alpha_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    return torch.clip(betas, 0, max_beta)


@register_beta('polynomial')
def get_polynomial_beta(beta_step, beta_start, beta_end, power=3, convex=True):
    x = torch.linspace(0, 1, beta_step, dtype=torch.float)
    if convex:
        return beta_start + (beta_end - beta_start) * x.pow(power)
    else:
        return beta_end - (beta_end - beta_start) * (1 - x).pow(power)


@register_beta('geometric')
def get_geometric_beta(beta_step, beta_start, beta_end):
    return torch.exp(torch.linspace(
        math.log(beta_start), math.log(beta_end),
        beta_step, dtype=torch.float
    ))
