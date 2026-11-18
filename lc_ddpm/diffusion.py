"""Gaussian diffusion: cosine schedule, v-prediction, DDIM sampling with classifier-free guidance.

A condition set is a dict ``{"survey", "cls", "num", "mask"}`` (see :mod:`lc_ddpm.data`):
survey/cls (B,) long, num (B, 3) normalised [log10 P, log10 A, mag], mask (B, 5) bool
(True = condition present). The unconditional pass sets the whole mask to False.
"""
import math

import torch
import torch.nn.functional as F


def cosine_alpha_bar(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


def clean_mask(cond):
    """A NaN magnitude always counts as missing."""
    mask = cond["mask"].clone()
    mask[:, 4] &= torch.isfinite(cond["num"][:, 2])
    return {**cond, "mask": mask, "num": torch.nan_to_num(cond["num"])}


def drop_conditions(cond, p=0.15, p_all=0.1, generator=None):
    """Training-time dropout: each condition independently with p, all jointly with p_all."""
    mask = cond["mask"]
    keep = torch.rand(mask.shape, device=mask.device, generator=generator) >= p
    keep &= torch.rand(mask.shape[0], 1, device=mask.device, generator=generator) >= p_all
    return {**cond, "mask": mask & keep}


class Diffusion:
    def __init__(self, T=1000):
        self.T = T
        self.alpha_bar = cosine_alpha_bar(T)

    def _ab(self, t, device):
        return self.alpha_bar.to(device)[t][:, None]

    def loss(self, model, x0, cond, weight=None, noise=None, t=None):
        """v-prediction MSE, optionally weighted per sample. x0: (B, nbins) in [-1, 1]."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self._ab(t, x0.device)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        cond = clean_mask(cond)
        err = F.mse_loss(model(xt, t, cond["survey"], cond["cls"], cond["num"], cond["mask"]).float(), v,
                         reduction="none").mean(1)
        return err.mean() if weight is None else (err * weight).sum() / weight.sum()

    @torch.no_grad()
    def sample(self, model, cond, steps=50, guidance=1.0, eta=0.0, generator=None, nbins=128):
        """DDIM sampling. guidance w: v = v_uncond + w (v_cond - v_uncond); w = 1 is plain conditional.

        Noise is drawn on the CPU from ``generator`` so a seed gives the same result on any device.
        Returns (B, nbins) in the model space [-1, 1].
        """
        device = cond["num"].device
        cond = clean_mask(cond)
        B = cond["num"].shape[0]
        randn = lambda: torch.randn(B, nbins, generator=generator).to(device)
        x = randn()
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        ab_all = self.alpha_bar.to(device)
        both = {k: torch.cat([v, v]) for k, v in cond.items()}
        both["mask"][B:] = False
        for i, t in enumerate(ts):
            tt = torch.full((2 * B,), t, device=device, dtype=torch.long)
            v = model(torch.cat([x, x]), tt, both["survey"], both["cls"], both["num"], both["mask"]).float()
            v = v[B:] + guidance * (v[:B] - v[B:])
            ab = ab_all[t]
            x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            if i == len(ts) - 1:
                x = x0
                break
            ab_prev = ab_all[ts[i + 1]]
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0:
                x = x + sigma * randn()
        return x
