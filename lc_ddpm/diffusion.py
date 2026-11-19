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
            x = ddim_step(x, v, ab_all[t], ab_all[ts[i + 1]] if i < len(ts) - 1 else None, eta, 1.0, randn)
        return x


def ddim_step(x, v, ab, ab_prev, eta=0.0, clip=1.0, randn=None):
    """One DDIM update from v-prediction ``v``; ``ab_prev=None`` returns the predicted x0 (last step).
    ``clip``: clamp x0 to [-clip, clip] (None = no clamp)."""
    x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
    if clip is not None:
        x0 = x0.clamp(-clip, clip)
    if ab_prev is None:
        return x0
    eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
    sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
    x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
    return x + sigma * randn() if eta > 0 else x


class SeqDiffusion(Diffusion):
    """The same diffusion over padded point sequences (v3, docs/DESIGN_v3.md).

    x0: (B, L, C). The condition dict also carries ``feat`` (B, L, F) per-point inputs and ``pad``
    (B, L) bool (True = real point); the model is called as ``model(x, t, cond)``. The loss is
    averaged over real points only and padded positions are kept at zero. ``clip`` (default None)
    clamps the predicted x0 during sampling; v3 channels are standardised, not bounded.
    """

    def loss(self, model, x0, cond, weight=None, noise=None, t=None):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        pad = cond["pad"][..., None].to(x0.dtype)
        ab = self.alpha_bar.to(x0.device)[t][:, None, None]
        xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * noise) * pad
        v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        err = ((model(xt, t, cond).float() - v) ** 2 * pad).sum((1, 2)) / (pad.sum((1, 2)) * x0.shape[2])
        return err.mean() if weight is None else (err * weight).sum() / weight.sum()

    @torch.no_grad()
    def sample(self, model, cond, channels, steps=50, guidance=1.0, eta=0.0, generator=None, clip=None):
        """DDIM with classifier-free guidance; returns (B, L, channels), zero at padded positions.
        Noise is drawn on the CPU from ``generator`` (same seed, same result on any device)."""
        pad = cond["pad"][..., None].float()
        B, L = cond["pad"].shape
        device = pad.device
        randn = lambda: torch.randn(B, L, channels, generator=generator).to(device) * pad
        x = randn()
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        ab_all = self.alpha_bar.to(device)
        both = {k: torch.cat([v, v]) for k, v in cond.items()}
        both["mask"][B:] = False
        for i, t in enumerate(ts):
            tt = torch.full((2 * B,), t, device=device, dtype=torch.long)
            v = model(torch.cat([x, x]), tt, both).float()
            v = v[B:] + guidance * (v[:B] - v[B:])
            x = ddim_step(x, v, ab_all[t], ab_all[ts[i + 1]] if i < len(ts) - 1 else None, eta, clip, randn) * pad
        return x
