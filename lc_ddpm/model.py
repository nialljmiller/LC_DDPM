"""1-D conditional U-Net for circular 128-point light-curve profiles."""
import math

import torch
from torch import nn
import torch.nn.functional as F

N_COND = 5  # mask columns: survey, class, log10 period, log10 amplitude, magnitude
N_NUM = 3   # numeric conditions: log10 period, log10 amplitude, magnitude (normalised)


def conv(cin, cout, k=3, stride=1):
    return nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular")


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, groups, dropout):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(groups, cin), conv(cin, cout)
        self.film = nn.Linear(emb_dim, 2 * cout)
        self.norm2, self.conv2 = nn.GroupNorm(groups, cout), conv(cout, cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(emb))[:, :, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(self.drop(F.silu(h)))


class Attention(nn.Module):
    def __init__(self, ch, groups, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        return x + self.attn(h, h, h, need_weights=False)[0].transpose(1, 2)


class CondEmbedding(nn.Module):
    """Sum of survey, class and numeric-condition embeddings; each falls back to a learned null."""

    def __init__(self, n_surveys, n_classes, dim):
        super().__init__()
        self.n_surveys, self.n_classes = n_surveys, n_classes
        self.survey = nn.Embedding(n_surveys + 1, dim)   # index n_surveys = null
        self.cls = nn.Embedding(n_classes + 1, dim)      # index n_classes = null
        self.num = nn.ModuleList(nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)) for _ in range(N_NUM))
        self.null = nn.Parameter(torch.zeros(N_NUM, dim))

    def forward(self, survey, cls, num, mask):
        """survey, cls: (B,) long; num: (B, 3) float; mask: (B, 5) bool, True = condition present."""
        e = self.survey(torch.where(mask[:, 0], survey, self.n_surveys))
        e = e + self.cls(torch.where(mask[:, 1], cls, self.n_classes))
        num = torch.nan_to_num(num)
        for j, mlp in enumerate(self.num):
            e = e + torch.where(mask[:, 2 + j, None], mlp(num[:, j:j + 1]), self.null[j])
        return e


class UNet1D(nn.Module):
    """U-Net over (B, 1, nbins) circular profiles, resolutions nbins / 2**level."""

    def __init__(self, n_surveys, n_classes, width=64, mults=(1, 2, 3, 4), n_res=2, attn_res=(32, 16),
                 groups=8, heads=4, dropout=0.0, nbins=128):
        super().__init__()
        self.config = dict(n_surveys=n_surveys, n_classes=n_classes, width=width, mults=tuple(mults), n_res=n_res,
                           attn_res=tuple(attn_res), groups=groups, heads=heads, dropout=dropout, nbins=nbins)
        emb_dim = 4 * width
        self.width = width
        self.time_mlp = nn.Sequential(nn.Linear(width, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond = CondEmbedding(n_surveys, n_classes, emb_dim)
        self.inp = conv(1, width)

        chans = [width * m for m in mults]
        self.down, skips, ch, res = nn.ModuleList(), [width], width, nbins
        for i, c in enumerate(chans):
            for _ in range(n_res):
                self.down.append(ResBlock(ch, c, emb_dim, groups, dropout)); ch = c
                if res in attn_res:
                    self.down.append(Attention(ch, groups, heads))
                skips.append(ch)
            if i < len(chans) - 1:
                self.down.append(conv(ch, ch, 3, stride=2)); res //= 2
                skips.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, groups, dropout), Attention(ch, groups, heads),
                                  ResBlock(ch, ch, emb_dim, groups, dropout)])
        self.up = nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            for _ in range(n_res + 1):
                self.up.append(ResBlock(ch + skips.pop(), c, emb_dim, groups, dropout)); ch = c
                if res in attn_res:
                    self.up.append(Attention(ch, groups, heads))
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv(ch, ch))); res *= 2
        self.out = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), conv(ch, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, survey, cls, num, mask):
        """x: (B, nbins) or (B, 1, nbins); t: (B,) diffusion step. Returns the same shape as x."""
        squeeze = x.dim() == 2
        h = self.inp(x[:, None] if squeeze else x)
        emb = self.time_mlp(timestep_embedding(t, self.width)) + self.cond(survey, cls, num, mask)
        hs = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            if isinstance(m, Attention):
                hs[-1] = h
            else:
                hs.append(h)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), emb)
            else:
                h = m(h)
        h = self.out(h)
        return h[:, 0] if squeeze else h
