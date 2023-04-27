from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import get_encoder
from .beta_schedule import get_beta


class DiffNet(nn.Module):
    def __init__(self, model_cfg, beta_schedule, time_embed_size=128, sample_scheme='ddpm',
                 clip_denoise=10., **kwargs):
        super(DiffNet, self).__init__()
        assert sample_scheme in ('ddpm', 'ddim', 'score')
        self.beta_schedule = beta_schedule
        self.beta_step = beta_schedule.beta_step
        self.use_score = sample_scheme == 'score'
        self.time_embed_size = time_embed_size
        self.sample_scheme = sample_scheme
        self.clip_denoise = clip_denoise

        self.model_cfg = model_cfg.copy()
        self.model_cfg.update(kwargs)
        self.model_cfg['time_embed_size'] = time_embed_size
        self.encoder = get_encoder(self.model_cfg)
        if self.sample_scheme == 'ddpm':
            self.sample_func = self.p_sample
        elif self.sample_scheme == 'ddim':
            self.sample_func = self.ddim_sample
        elif self.sample_scheme == 'score':
            self.sample_func = self.score_sample
        else:
            raise ValueError(f'Unknown sample scheme {sample_scheme}')

        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alpha_cumprod: torch.Tensor
        self.alpha_cumprod_prev: torch.Tensor
        self.sqrt_alpha_cumprod: torch.Tensor
        self.sqrt_one_minus_alpha_cumprod: torch.Tensor
        self.sqrt_recip_alpha_cumprod: torch.Tensor
        self.sqrt_recipm1_alpha_cumprod: torch.Tensor
        self.posterior_variance: torch.Tensor
        self.posterior_log_variance: torch.Tensor
        self.posterior_mean_coef1: torch.Tensor
        self.posterior_mean_coef2: torch.Tensor

        betas = get_beta(beta_schedule)
        self.register_buffer('betas', betas)

        alphas = 1. - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.)
        posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod).unsqueeze(-1))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1. - alpha_cumprod).unsqueeze(-1))
        self.register_buffer('sqrt_recip_alpha_cumprod', torch.sqrt(1. / alpha_cumprod))
        self.register_buffer('sqrt_recipm1_alpha_cumprod', torch.sqrt(1. / alpha_cumprod - 1.))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod))
        self.register_buffer(
            'posterior_mean_coef2', (1. - alpha_cumprod_prev) * torch.sqrt(alphas) / (1. - alpha_cumprod)
        )

    @torch.no_grad()
    def p_sample(self, atom_types, atom_coord, batch, init_feat=None, gt=None, add_noise=False):
        node_feat = torch.randn(batch.size(0), self.encoder.n_gauss, device=batch.device) \
            if init_feat is None else init_feat
        preds = [node_feat]
        for t in tqdm(reversed(range(self.beta_step)), total=self.beta_step):
            timestep = torch.full_like(batch, t, dtype=torch.long)
            pred_noise = self.encoder(atom_types, atom_coord, batch, node_feat, timestep)
            if gt is not None:
                pred_noise = (node_feat - self.sqrt_alpha_cumprod[t] * gt) / self.sqrt_one_minus_alpha_cumprod[t]
            pred_start = (
                    node_feat * self.sqrt_recip_alpha_cumprod[t] -
                    pred_noise * self.sqrt_recipm1_alpha_cumprod[t]
            )
            posterior_mean = (
                    pred_start * self.posterior_mean_coef1[t] +
                    node_feat * self.posterior_mean_coef2[t]
            )
            noise = torch.randn_like(node_feat) if add_noise and t > 0 else 0.
            node_feat = posterior_mean + (0.5 * self.posterior_log_variance[t]).exp() * noise
            node_feat = node_feat.clamp(-self.clip_denoise, self.clip_denoise)
            preds.append(node_feat)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def ddim_sample(self, atom_types, atom_coord, batch, init_feat=None, gt=None):
        node_feat = torch.randn(batch.size(0), self.encoder.n_gauss, device=batch.device) \
            if init_feat is None else init_feat
        preds = [node_feat]
        for t in tqdm(reversed(range(self.beta_step)), total=self.beta_step):
            timestep = torch.full_like(batch, t, dtype=torch.long)
            pred_noise = self.encoder(atom_types, atom_coord, batch, node_feat, timestep)
            if gt is not None:
                pred_noise = (node_feat - self.sqrt_alpha_cumprod[t] * gt) / self.sqrt_one_minus_alpha_cumprod[t]
            pred_start = (
                    node_feat * self.sqrt_recip_alpha_cumprod[t] -
                    pred_noise * self.sqrt_recipm1_alpha_cumprod[t]
            )
            alpha_next = self.alpha_cumprod_prev[t]
            node_feat = (
                    pred_start * alpha_next.sqrt() +
                    pred_noise * (1 - alpha_next).sqrt()
            )
            node_feat = node_feat.clamp(-self.clip_denoise, self.clip_denoise)
            preds.append(node_feat)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def score_sample(self, atom_types, atom_coord, batch, sigmas=None, n_steps_each=10, step_lr=2e-6,
                     init_feat=None, gt=None, add_noise=False):
        sigmas = self.betas if sigmas is None else sigmas
        node_feat = torch.randn(batch.size(0), self.encoder.n_gauss, device=batch.device) \
            if init_feat is None else init_feat
        preds = [node_feat]
        for t, sigma in tqdm(reversed(list(enumerate(sigmas))), total=sigmas.size(0)):
            step_size = step_lr * (sigma / sigmas[0]) ** 2
            for _ in range(n_steps_each):
                timestep = torch.full_like(batch, t, dtype=torch.long)
                pred_noise = self.encoder(atom_types, atom_coord, batch, node_feat, timestep) / sigma
                if gt is not None:
                    pred_noise = (gt - node_feat) / sigma ** 2
                noise = torch.randn_like(node_feat) * torch.sqrt(step_size * 2) if add_noise else 0.
                node_feat = node_feat + step_size * pred_noise + noise
                node_feat = node_feat.clamp(-self.clip_denoise, self.clip_denoise)
                preds.append(node_feat)
        return torch.stack(preds, dim=1)

    def sample(self, atom_types, atom_coord, batch, **kwargs):
        return self.sample_func(atom_types, atom_coord, batch, **kwargs)

    def forward(self, node_feat, atom_types, atom_coord, batch):
        t = torch.randint_like(batch, 0, self.beta_step)
        noise = torch.randn_like(node_feat)
        sigmas = self.betas[t].unsqueeze(-1)
        if self.use_score:
            noise = noise * sigmas
            diffused = node_feat + noise
        else:
            diffused = self.sqrt_alpha_cumprod[t] * node_feat + self.sqrt_one_minus_alpha_cumprod[t] * noise

        score = self.encoder(atom_types, atom_coord, batch, node_feat=diffused, timestep=t)
        if self.use_score:
            score = score / sigmas
            target = -noise / (sigmas ** 2)
            loss = 0.5 * ((score - target) ** 2).sum(-1) * (sigmas.squeeze(-1) ** 2)
            loss = loss.mean(0)
        else:
            loss = F.mse_loss(score, noise)
            # loss = F.l1_loss(score, noise)
        return loss
