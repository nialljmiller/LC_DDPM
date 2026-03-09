"""
Stable Autonomous Flow Matching for Light Curves — Conditional on PRIMVS Features.

Based on: "Stable Autonomous Flow Matching" (Sprague et al., 2024)
  arXiv:2402.05774

Conditioning: the model is conditioned on a vector of PRIMVS catalog features
(period, amplitude, mean_mag, skew, etc.).  At inference time the user supplies
those values and the model generates a light curve consistent with them.

Channel layout of each training sample  (3, spatial_size, spatial_size):
  0 : magnitude       (tiled)
  1 : magnitude error (tiled)
  2 : normalised time  (mjd - mjd_min) / mjd_range in [0,1]  (tiled)
"""

import json
import math
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from primvs_pipeline import primvs_api as api
PrimvsCatalog = api.PrimvsCatalog


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default PRIMVS catalog columns used as conditioning features.
# Any column present in the FITS table can be added here.
# Columns whose name contains "period" are log-transformed automatically.
DEFAULT_COND_FEATURES = [
    "true_period",    # dominant variability period  (log-transformed)
    "amplitude",      # peak-to-peak amplitude in Ks
    "mean_mag",       # mean Ks magnitude
    "skewness",       # flux distribution skewness
    "kurtosis",       # flux distribution kurtosis
    "ad_pvalue",      # Anderson-Darling p-value for periodicity
    "stetson_k",      # Stetson-K variability index
    "color_jk",       # J - Ks mean colour
    "color_hk",       # H - Ks mean colour
]


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def num_to_groups(num, divisor):
    groups    = num // divisor
    remainder = num % divisor
    arr       = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _delete_rand_items_3col(a, b, c, n):
    keep = np.sort(np.random.choice(len(a), len(a) - n, replace=False))
    return a[keep], b[keep], c[keep]


def tile_to_square(arr, spatial_size):
    n    = spatial_size * spatial_size
    reps = math.ceil(n / len(arr))
    return np.tile(arr, reps)[:n].reshape(spatial_size, spatial_size)


# --------------------------------------------------------------------------- #
#  Sample visualisation                                                        #
# --------------------------------------------------------------------------- #

def _save_sample_lightcurves(samples, step, logdir,
                              cond_tensors=None, feature_names=None,
                              feature_stats=None, ncols=6):
    """
    Save two grids of light curve plots for a batch of generated samples:
      {step}-unfolded.png     — magnitude vs sequential index
      {step}-phasefolded.png  — magnitude vs phase channel (sorted)

    When cond_tensors + feature_stats are provided, period and amplitude
    are denormalised and printed as a subtitle on each subplot.
    """
    samples_np = samples.detach().cpu().numpy()   # (N, 3, S, S)
    N    = len(samples_np)
    nrows = math.ceil(N / ncols)

    fig_u, axes_u = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8), squeeze=False)
    fig_p, axes_p = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8), squeeze=False)

    for ax in axes_u.flatten():
        ax.set_visible(False)
    for ax in axes_p.flatten():
        ax.set_visible(False)

    # Denormalise conditioning features for plot annotation
    cond_denorm = None
    if cond_tensors is not None and feature_stats and feature_names:
        cond_np     = cond_tensors.detach().cpu().numpy()
        means       = np.array(feature_stats["mean"])
        stds        = np.array(feature_stats["std"])
        cond_denorm = cond_np * stds + means
        for j, fname in enumerate(feature_names):
            if "period" in fname.lower():
                cond_denorm[:, j] = np.expm1(cond_denorm[:, j])

    for i, sample in enumerate(samples_np):
        mag   = sample[0].flatten().astype(np.float64)
        err   = np.abs(sample[1].flatten().astype(np.float64))
        phase = sample[2].flatten().astype(np.float64)

        stride     = max(1, len(mag) // 300)
        idx        = np.arange(0, len(mag), stride)
        mag_s      = mag[idx]
        err_s      = err[idx]
        phase_s    = phase[idx]

        # Build subtitle: show period + amplitude if available
        subtitle = ""
        if cond_denorm is not None and feature_names:
            parts = []
            for fname in ("true_period", "amplitude"):
                if fname in feature_names:
                    j = feature_names.index(fname)
                    parts.append(f"{fname[:3]}={cond_denorm[i, j]:.2f}")
            subtitle = "  ".join(parts)
        title = f"#{i}" + (f"\n{subtitle}" if subtitle else "")

        ax = axes_u.flatten()[i]
        ax.set_visible(True)
        ax.errorbar(np.arange(len(mag_s)), mag_s, yerr=err_s,
                    fmt="o", ms=2, alpha=0.6,
                    color="steelblue", ecolor="lightsteelblue", elinewidth=0.8, capsize=0)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel("Index", fontsize=5)
        ax.set_ylabel("Mag", fontsize=5)
        ax.tick_params(labelsize=4)

        ax = axes_p.flatten()[i]
        ax.set_visible(True)
        order = np.argsort(phase_s)
        ax.errorbar(phase_s[order], mag_s[order], yerr=err_s[order],
                    fmt="o", ms=2, alpha=0.6,
                    color="tomato", ecolor="lightsalmon", elinewidth=0.8, capsize=0)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel("Phase", fontsize=5)
        ax.set_ylabel("Mag", fontsize=5)
        ax.tick_params(labelsize=4)

    fig_u.suptitle(f"Step {step:,} — Unfolded", fontsize=10, y=1.01)
    fig_p.suptitle(f"Step {step:,} — Phase-Folded", fontsize=10, y=1.01)
    for fig in (fig_u, fig_p):
        fig.tight_layout()

    path_u = logdir / f"{step:08d}-unfolded.png"
    path_p = logdir / f"{step:08d}-phasefolded.png"
    fig_u.savefig(str(path_u), dpi=120, bbox_inches="tight")
    fig_p.savefig(str(path_p), dpi=120, bbox_inches="tight")
    plt.close(fig_u)
    plt.close(fig_p)
    print(f"  saved {path_u.name}  &  {path_p.name}")


# --------------------------------------------------------------------------- #
#  EMA                                                                         #
# --------------------------------------------------------------------------- #

class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for cp, mp in zip(current_model.parameters(), ma_model.parameters()):
            mp.data = mp.data * self.beta + (1 - self.beta) * cp.data


# --------------------------------------------------------------------------- #
#  Small modules                                                               #
# --------------------------------------------------------------------------- #

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
#  Building blocks                                                             #
# --------------------------------------------------------------------------- #

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=16):
        super().__init__()
        self.mlp      = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1   = Block(dim,     dim_out, groups=groups)
        self.block2   = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h  = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h  = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads  = heads
        hidden_dim  = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv        = self.to_qkv(x)
        q, k, v    = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)",
                                heads=self.heads, qkv=3)
        k          = k.softmax(dim=-1)
        context    = torch.einsum("bhdn,bhen->bhde", k, v)
        out        = torch.einsum("bhde,bhdn->bhen", context, q)
        out        = rearrange(out, "b heads c (h w) -> b (heads c) h w",
                               heads=self.heads, h=h, w=w)
        return self.to_out(out)


# --------------------------------------------------------------------------- #
#  U-Net                                                                      #
# --------------------------------------------------------------------------- #

class Unet(nn.Module):
    """
    U-Net predicting the vector field v_θ(z, τ | c).

    Conditioning features c are projected to `dim` and *added* to the
    sinusoidal time embedding before being broadcast into every ResNet block,
    the same way the time signal is used.

    When num_cond_features=0 the model is fully unconditional.
    """

    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        groups=12,
        channels=3,
        num_cond_features=0,
    ):
        super().__init__()
        self.channels          = channels
        self.num_cond_features = num_cond_features

        dims   = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ---- time embedding
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp     = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # ---- conditioning MLP
        if num_cond_features > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(num_cond_features, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim),
            )
        else:
            self.cond_mlp = None

        # ---- U-Net layers
        self.downs      = nn.ModuleList([])
        self.ups        = nn.ModuleList([])
        num_res         = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_res - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in,  d_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(d_out, d_out, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention(d_out))),
                Downsample(d_out) if not is_last else nn.Identity(),
            ]))

        mid_dim         = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn   = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_res - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_out * 2, d_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(d_in,      d_in, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention(d_in))),
                Upsample(d_in) if not is_last else nn.Identity(),
            ]))

        out_dim_         = default(out_dim, channels)
        self.final_conv  = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim_, 1),
        )

    def forward(self, x, tau, cond=None):
        """
        Args:
            x    : (B, C, H, W)
            tau  : (B,)           pseudo-time in [0, 1]
            cond : (B, F)         normalised conditioning features, or None
        """
        t = self.time_mlp(self.time_pos_emb(tau * 1000.0))

        if self.cond_mlp is not None and cond is not None:
            t = t + self.cond_mlp(cond)

        h = []
        for resnet, resnet2, attn, down in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = up(x)

        return self.final_conv(x)


# --------------------------------------------------------------------------- #
#  Stable Flow Matching                                                        #
# --------------------------------------------------------------------------- #

class StableFlowMatching(nn.Module):
    """Stable Autonomous Flow Matching, conditioned on catalog feature vectors."""

    def __init__(
        self,
        vector_field_fn,
        *,
        spatial_size=128,
        channels=3,
        lambda_z=2.0,
        lambda_tau=1.0,
        num_sample_steps=100,
        loss_type="l2",
    ):
        super().__init__()
        self.vector_field_fn  = vector_field_fn
        self.spatial_size     = spatial_size
        self.channels         = channels
        self.lambda_z         = lambda_z
        self.lambda_tau       = lambda_tau
        self.ratio            = lambda_z / lambda_tau
        self.num_sample_steps = num_sample_steps
        self.loss_type        = loss_type

    def _interpolate(self, z1, tau):
        z0    = torch.randn_like(z1)
        alpha = (1.0 - tau).pow(self.ratio).view(-1, 1, 1, 1)
        return z1 + alpha * (z0 - z1), z0

    def _target(self, z_tau, z1):
        return -self.lambda_z * (z_tau - z1)

    def forward(self, z1, cond=None):
        b       = z1.shape[0]
        device  = z1.device
        tau     = torch.rand(b, device=device)
        z_tau, _ = self._interpolate(z1, tau)
        target   = self._target(z_tau, z1)
        pred     = self.vector_field_fn(z_tau, tau, cond=cond)
        if self.loss_type == "l1":
            return (pred - target).abs().mean()
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target)
        else:
            raise NotImplementedError(self.loss_type)

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, num_steps=None):
        """
        Generate light curves.
        cond : (batch_size, F) normalised feature tensor, or None for unconditional.
        """
        num_steps  = num_steps or self.num_sample_steps
        device     = next(self.vector_field_fn.parameters()).device
        S          = self.spatial_size
        z          = torch.randn(batch_size, self.channels, S, S, device=device)
        T          = 5.0 / self.lambda_tau
        dt         = T / num_steps
        t_vals     = torch.linspace(0, T, num_steps + 1, device=device)
        tau_vals   = 1.0 - torch.exp(-self.lambda_tau * t_vals)
        for i in tqdm(range(num_steps), desc="sampling", leave=False):
            tau   = tau_vals[i].expand(batch_size)
            z     = z + self.vector_field_fn(z, tau, cond=cond) * dt
        return z

    @torch.no_grad()
    def sample_midpoint(self, batch_size=16, cond=None, num_steps=None):
        num_steps  = num_steps or self.num_sample_steps
        device     = next(self.vector_field_fn.parameters()).device
        S          = self.spatial_size
        z          = torch.randn(batch_size, self.channels, S, S, device=device)
        T          = 5.0 / self.lambda_tau
        dt         = T / num_steps
        t_vals     = torch.linspace(0, T, num_steps + 1, device=device)
        tau_vals   = 1.0 - torch.exp(-self.lambda_tau * t_vals)
        for i in tqdm(range(num_steps), desc="sampling (midpoint)", leave=False):
            tau_i   = tau_vals[i].expand(batch_size)
            tau_mid = (0.5 * (tau_vals[i] + tau_vals[i + 1])).expand(batch_size)
            v1      = self.vector_field_fn(z, tau_i, cond=cond)
            z_mid   = z + v1 * (dt / 2.0)
            v2      = self.vector_field_fn(z_mid, tau_mid, cond=cond)
            z       = z + v2 * dt
        return z


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LazyPrimvsDataset(data.Dataset):
    """
    Lazily reads individual light curve CSVs on demand.
    Also loads conditioning features from a PRIMVS catalog FITS file.

    __getitem__ always returns (lc_tensor, cond_tensor).
    cond_tensor is zero-length when no features are configured.
    """

    def __init__(
        self,
        source_ids,
        data_dir,
        lc_size=70,
        spatial_size=128,
        band="Ks",
        catalog_fits=None,
        cond_feature_names=None,
    ):
        super().__init__()
        self.cat                = PrimvsCatalog(data_dir)
        self.lc_size            = lc_size
        self.spatial_size       = spatial_size
        self.band               = band
        self.cond_feature_names = list(cond_feature_names) if cond_feature_names else []
        self.feature_lookup     = {}   # int(source_id) → float32 array (normalised)
        self.feature_stats      = {}   # persisted to JSON for inference

        self.source_ids = [int(sid) for sid in source_ids
                          if self.cat.source_exists(sid)]
        print(f"LazyPrimvsDataset: {len(self.source_ids)} sources on disk "
              f"(band={band}, lc_size={lc_size}, spatial={spatial_size}x{spatial_size})")

        if catalog_fits and self.cond_feature_names:
            self._load_catalog_features(catalog_fits, self.cond_feature_names)

    # ------------------------------------------------------------------
    def _load_catalog_features(self, catalog_fits, feature_names):
        from astropy.table import Table
        tbl       = Table.read(catalog_fits, hdu=1)
        available = [f for f in feature_names if f in tbl.colnames]
        missing   = [f for f in feature_names if f not in tbl.colnames]
        if missing:
            print(f"  Warning — features not in catalog (will be 0): {missing}")

        ids     = np.array(tbl["sourceid"].data, dtype=np.int64)
        F       = len(feature_names)
        raw     = np.full((len(ids), F), np.nan, dtype=np.float64)

        for j, fname in enumerate(feature_names):
            if fname in tbl.colnames:
                col = np.array(tbl[fname].data, dtype=np.float64)
                if "period" in fname.lower():
                    col = np.log1p(np.abs(col))
                raw[:, j] = col

        means         = np.nanmean(raw, axis=0)
        stds          = np.nanstd(raw,  axis=0)
        stds[stds == 0] = 1.0

        self.feature_stats = {
            "names": feature_names,
            "mean":  means.tolist(),
            "std":   stds.tolist(),
        }

        norm = np.nan_to_num((raw - means) / stds, nan=0.0).astype(np.float32)
        for i, sid in enumerate(ids):
            self.feature_lookup[int(sid)] = norm[i]

        print(f"  Loaded conditioning features for {len(self.feature_lookup)} catalog sources")
        print(f"  Features ({F}): {feature_names}")

    # ------------------------------------------------------------------
    def num_cond_features(self):
        return len(self.cond_feature_names)

    def get_random_cond_batch(self, n, device="cpu"):
        """Return (n, F) tensor of randomly sampled normalised feature vectors."""
        if not self.feature_lookup:
            return None
        keys   = list(self.feature_lookup.keys())
        chosen = np.random.choice(len(keys), size=min(n, len(keys)), replace=False)
        vecs   = np.stack([self.feature_lookup[keys[i]] for i in chosen], axis=0)
        return torch.from_numpy(vecs).to(device)

    def save_feature_stats(self, path):
        with open(path, "w") as f:
            json.dump(self.feature_stats, f, indent=2)
        print(f"  Feature stats saved → {path}")

    # ------------------------------------------------------------------
    def _process(self, sid):
        df = self.cat.get_lightcurve(sid)
        if df is None:
            return None
        sub = df[df["filter"] == self.band].dropna(subset=["mag", "err"]).copy()
        if len(sub) < self.lc_size:
            return None
        mjd = sub["mjd"].values.astype(np.float32)
        mag = sub["mag"].values.astype(np.float32)
        err = sub["err"].values.astype(np.float32)
        order          = np.argsort(mjd)
        mjd, mag, err  = mjd[order], mag[order], err[order]
        if len(mag) > self.lc_size:
            mjd, mag, err = _delete_rand_items_3col(mjd, mag, err, len(mag) - self.lc_size)
        if np.any(np.isnan(mag)) or np.any(np.isnan(err)):
            return None
        rng = mjd.max() - mjd.min()
        if rng == 0:
            return None
        phase = (mjd - mjd.min()) / rng
        s     = self.spatial_size
        return np.stack([
            tile_to_square(mag,   s),
            tile_to_square(err,   s),
            tile_to_square(phase, s),
        ], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        sid = self.source_ids[index]
        seq = self._process(sid)
        if seq is None:
            sid = self.source_ids[(index + 1) % len(self.source_ids)]
            seq = self._process(sid)
        if seq is None:
            seq = np.zeros((3, self.spatial_size, self.spatial_size), dtype=np.float32)

        tensor = torch.from_numpy(seq)
        if self.cond_feature_names and self.feature_lookup:
            raw  = self.feature_lookup.get(
                int(sid),
                np.zeros(len(self.cond_feature_names), dtype=np.float32),
            )
            cond = torch.from_numpy(raw.copy())
        else:
            cond = torch.zeros(0, dtype=torch.float32)

        return tensor, cond


class LightcurveDataset(data.Dataset):
    """Thin wrapper used by infer.py to load a checkpoint via Trainer."""

    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
#  Trainer                                                                     #
# --------------------------------------------------------------------------- #

class Trainer:
    def __init__(
        self,
        flow_model,
        dataset,
        *,
        ema_decay=0.995,
        spatial_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        rank=None,
        num_workers=8,
        save_every=5000,
        sample_every=5000,
        logdir="./logs",
    ):
        rank = rank or [0]

        self.model              = torch.nn.DataParallel(flow_model, device_ids=rank)
        self.ema                = EMA(ema_decay)
        self.ema_model          = copy.deepcopy(self.model)
        self.update_ema_every   = update_ema_every
        self.step_start_ema     = step_start_ema
        self.save_every         = save_every
        self.sample_every       = sample_every
        self.batch_size         = train_batch_size
        self.spatial_size       = spatial_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps    = train_num_steps

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.ds = dataset
        self.dl = cycle(data.DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
        ))

        self.opt  = Adam(flow_model.parameters(), lr=train_lr)
        self.step = 0
        self.reset_parameters()

        # Persist feature stats once at startup so infer.py can reload them
        if hasattr(dataset, "feature_stats") and dataset.feature_stats:
            dataset.save_feature_stats(self.logdir / "feature_stats.json")

    # ------------------------------------------------------------------
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        torch.save(
            {"step": self.step,
             "model": self.model.state_dict(),
             "ema":   self.ema_model.state_dict()},
            str(self.logdir / f"{milestone:08d}-model.pt"),
        )

    def load(self, milestone):
        p = torch.load(str(self.logdir / f"{milestone:08d}-model.pt"), map_location="cpu")
        self.step = p["step"]
        self.model.load_state_dict(p["model"])
        self.ema_model.load_state_dict(p["ema"])

    # ------------------------------------------------------------------
    def _to_cond(self, cond_batch):
        """Move cond to device, return None if it carries no features."""
        if cond_batch is None:
            return None
        if cond_batch.numel() == 0 or cond_batch.shape[-1] == 0:
            return None
        return cond_batch.to(device=DEVICE, dtype=torch.float)

    def train(self):
        feature_names = getattr(self.ds, "cond_feature_names", None) or None
        feature_stats = getattr(self.ds, "feature_stats",      None) or None

        t1 = time()
        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                lc, cond = next(self.dl)
                lc       = lc.to(device=DEVICE, dtype=torch.float)
                cond     = self._to_cond(cond)

                loss = self.model(lc, cond).sum()
                t0   = time()
                print(f"{self.step}: {loss.item():.6f}  Δt: {t0 - t1:.3f}s")
                t1   = time()
                with open(str(self.logdir / "loss.txt"), "a") as f:
                    f.write(f"{self.step},{loss.item()}\n")
                (loss / self.gradient_accumulate_every).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.sample_every == 0:
                n_samples   = 18
                batches     = num_to_groups(n_samples, self.batch_size)

                # Use random real conditioning vectors so sample plots are
                # annotated with actual catalog values
                sample_cond = None
                if hasattr(self.ds, "get_random_cond_batch"):
                    sample_cond = self.ds.get_random_cond_batch(n_samples, device=DEVICE)

                all_samples = []
                offset      = 0
                for n in batches:
                    c = sample_cond[offset:offset + n] if sample_cond is not None else None
                    all_samples.append(self.ema_model.module.sample(batch_size=n, cond=c))
                    offset += n
                all_samples = torch.cat(all_samples, dim=0)

                _save_sample_lightcurves(
                    all_samples, self.step, self.logdir,
                    cond_tensors=sample_cond,
                    feature_names=feature_names,
                    feature_stats=feature_stats,
                )

            if self.step != 0 and self.step % self.save_every == 0:
                self.save(self.step)

            self.step += 1

        print("Training completed.")
