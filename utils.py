import random
import resource

import yaml
from easydict import EasyDict
import numpy as np
import torch


def extend_ulimit():
    """
    Increase the maximum number of open file descriptors
    This bug may or may not occur depending on the running environment
    See https://github.com/pytorch/pytorch/issues/973 for more details
    Another solution is to set `torch.multiprocessing.set_sharing_strategy('file_system')`,
    :return: none
    """
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed):
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError(f'Optimizer not supported: {cfg.type}')


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    else:
        raise NotImplementedError(f'Scheduler not supported: {cfg.type}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
