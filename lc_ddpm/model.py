"""Networks: 1-D conditional U-Net for circular 128-point profiles (v2) and a transformer over
irregularly sampled points (v3, :class:`PointTransformer`)."""
import math

import torch
import torch.utils.checkpoint
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

    def __init__(self, n_surveys, n_classes, dim, n_num=N_NUM):
        super().__init__()
        self.n_surveys, self.n_classes = n_surveys, n_classes
        self.survey = nn.Embedding(n_surveys + 1, dim)   # index n_surveys = null
        self.cls = nn.Embedding(n_classes + 1, dim)      # index n_classes = null
        self.num = nn.ModuleList(nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)) for _ in range(n_num))
        self.null = nn.Parameter(torch.zeros(n_num, dim))

    def forward(self, survey, cls, num, mask):
        """survey, cls: (B,) long; num: (B, n_num) float; mask: (B, 2 + n_num) bool, True = present."""
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


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning (shift, scale and gate from the embedding)."""

    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, h, emb, attn_mask):
        B, L, D = h.shape
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(emb))[:, None].chunk(6, dim=-1)
        q, k, v = self.qkv(modulate(self.norm1(h), b1, s1)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h + g1 * self.proj(y.transpose(1, 2).reshape(B, L, D))
        return h + g2 * self.mlp(modulate(self.norm2(h), b2, s2))


class PointTransformer(nn.Module):
    """Transformer over a padded set of points (B, L, in_ch) with per-point inputs (B, L, feat_dim).

    Conditioning: diffusion-step embedding + :class:`CondEmbedding` (survey, class, ``n_num`` numeric
    conditions), applied through adaLN in every block. Attention masks padded keys only, so every
    query row has at least one valid key; outputs at padded positions are zero.
    ``gate = (mask_col, start, stop)``: per-point input columns [start, stop) are zeroed when
    condition column ``mask_col`` is missing (v3: the phase features follow the period condition).
    ``n_classes = 0``: no class condition (the class embedding only has its null entry).
    """

    def __init__(self, in_ch, feat_dim, n_surveys, n_classes, n_num, width=256, depth=8, heads=8, mlp_ratio=4,
                 dropout=0.0, gate=None, grad_ckpt=False):
        super().__init__()
        self.config = dict(in_ch=in_ch, feat_dim=feat_dim, n_surveys=n_surveys, n_classes=n_classes, n_num=n_num,
                           width=width, depth=depth, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout,
                           gate=tuple(gate) if gate else None)
        self.grad_ckpt, self.width = grad_ckpt, width
        self.inp = nn.Linear(in_ch + feat_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.cond = CondEmbedding(n_surveys, n_classes, width, n_num)
        self.blocks = nn.ModuleList(Block(width, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, in_ch)
        for m in (self.ada, self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, cond):
        """x: (B, L, in_ch); t: (B,); cond: survey, cls (B,), num (B, n_num), mask (B, 2 + n_num),
        feat (B, L, feat_dim), pad (B, L) bool. Returns (B, L, in_ch)."""
        feat, pad, mask = cond["feat"], cond["pad"], cond["mask"]
        if self.config["n_classes"] == 0:
            mask = torch.cat([mask[:, :1], torch.zeros_like(mask[:, 1:2]), mask[:, 2:]], 1)
        if self.config["gate"]:
            col, a, b = self.config["gate"]
            on = mask[:, col].to(feat.dtype)[:, None, None]
            feat = torch.cat([feat[..., :a], feat[..., a:b] * on, feat[..., b:]], -1)
        h = self.inp(torch.cat([x, feat.to(x.dtype)], -1))
        emb = self.time_mlp(timestep_embedding(t, self.width)) + self.cond(cond["survey"], cond["cls"], cond["num"], mask)
        attn_mask = pad[:, None, None, :]
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, emb, attn_mask, use_reentrant=False)
            else:
                h = blk(h, emb, attn_mask)
        shift, scale = self.ada(F.silu(emb))[:, None].chunk(2, dim=-1)
        return self.out(modulate(self.norm(h), shift, scale)) * pad[..., None]
