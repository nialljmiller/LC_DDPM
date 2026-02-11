"""
Stable Autonomous Flow Matching for Light Curves.

Based on: "Stable Autonomous Flow Matching" (Sprague et al., 2024)
  arXiv:2402.05774

Replaces DDPM with the Stable-FM framework:
  - Augmented state (z, τ) with pseudo-time τ ∈ [0, 1]
  - Scalar stable CCNF: v'(x|x') = [-λ_z(z-z'), -λ_τ(τ-1)]
  - Unnormalized auto CFM loss L'_Auto (Definition 4.6)
  - λ_z/λ_τ controls interpolation rate (Lemma 4.11, Figure 3)
  - λ_z/λ_τ = 1 recovers OT-FM (Corollary 4.12)
"""

import math
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.optim import Adam
from torchvision import utils
from functools import partial
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from time import time

from primvs_api import PrimvsCatalog

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _delete_rand_items_3col(arr1, arr2, arr3, n):
    """Randomly remove n items from three aligned arrays."""
    indices = random.sample(range(len(arr1)), n)
    mask = np.ones(len(arr1), dtype=bool)
    mask[indices] = False
    return arr1[mask], arr2[mask], arr3[mask]


def tile_to_square(arr, size):
    """
    Tile a 1D array into a (size, size) 2D array.

    The array is repeated as many times as needed to fill size*size elements,
    then reshaped. This gives the UNet a spatially consistent input.

    Args:
        arr: 1D numpy array
        size: target spatial dimension (both H and W)
    Returns:
        (size, size) numpy array
    """
    total = size * size
    reps = int(np.ceil(total / len(arr)))
    tiled = np.tile(arr, reps)[:total]
    return tiled.reshape(size, size)


# --------------------------------------------------------------------------- #
#  PRIMVS data loading                                                         #
# --------------------------------------------------------------------------- #

def load_sequences_from_primvs(source_ids, data_dir, lc_size, spatial_size=128, band='Ks'):
    """
    Load lightcurve sequences from the PRIMVS catalog via PrimvsCatalog.

    Args:
        source_ids: list/array of VIRAC source IDs.
        data_dir: path to the PRIMVS lightcurve data directory.
        lc_size: minimum number of datapoints required per lightcurve.
        spatial_size: H and W of the output 2D representation (must be divisible by 16).
        band: photometric band to use (default 'Ks').

    Returns:
        list of numpy arrays, each shaped (3, spatial_size, spatial_size).
        Channels are [mag, err, phase] where phase = (mjd - mjd_min) / (mjd_max - mjd_min).
    """
    cat = PrimvsCatalog(data_dir)
    results = cat.get_lightcurves(source_ids)

    sequences = []
    for source_id, df in tqdm(results.items(), desc='Loading PRIMVS lightcurves'):
        # Filter to requested band
        subset = df[df['filter'] == band].dropna(subset=['mag', 'err']).copy()
        if len(subset) < 2:
            continue

        mjd = subset['mjd'].values.astype(float)
        mag = subset['mag'].values.astype(float)
        err = subset['err'].values.astype(float)

        # Sort by time
        order = np.argsort(mjd)
        mjd, mag, err = mjd[order], mag[order], err[order]

        # Skip if not enough points
        if len(mag) < lc_size:
            continue

        # Trim to lc_size by randomly removing excess points
        if len(mag) > lc_size:
            mjd, mag, err = _delete_rand_items_3col(mjd, mag, err, len(mag) - lc_size)

        # Skip any NaN sequences
        if np.any(np.isnan(mag)) or np.any(np.isnan(err)) or np.any(np.isnan(mjd)):
            continue

        # Compute phase from MJD
        mjd_range = mjd.max() - mjd.min()
        if mjd_range == 0:
            continue
        phase = (mjd - mjd.min()) / mjd_range

        # Tile each channel into a (spatial_size, spatial_size) 2D array
        sequence = np.stack((
            tile_to_square(mag, spatial_size),
            tile_to_square(err, spatial_size),
            tile_to_square(phase, spatial_size),
        ), axis=0)

        sequences.append(sequence)

    print(f"Loaded {len(sequences)} valid sequences from PRIMVS catalog")
    return sequences


# --------------------------------------------------------------------------- #
#  EMA                                                                         #
# --------------------------------------------------------------------------- #

class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * self.beta + (1 - self.beta) * up_weight


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
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

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
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads, qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


# --------------------------------------------------------------------------- #
#  U-Net (vector field predictor)                                              #
# --------------------------------------------------------------------------- #

class Unet(nn.Module):
    """
    U-Net that predicts the vector field v_θ(z, τ).

    Takes (z, τ) and outputs a vector field of the same spatial shape as z.
    τ is embedded via sinusoidal positional encoding (scaled to [0, 1000]
    for comparable frequency range to DDPM integer timesteps).
    """

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), groups=12, channels=3):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1),
        )

    def forward(self, x, tau):
        """
        Args:
            x: (B, C, H, W) spatial state z
            tau: (B,) pseudo-time in [0, 1]
        Returns:
            (B, C, H, W) predicted vector field for z
        """
        # Scale tau to [0, 1000] for sinusoidal embedding frequency range
        t = self.mlp(self.time_pos_emb(tau * 1000.0))
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# --------------------------------------------------------------------------- #
#  Stable Flow Matching                                                        #
# --------------------------------------------------------------------------- #

class StableFlowMatching(nn.Module):
    """
    Stable Autonomous Flow Matching (Sprague et al., 2024).

    Replaces GaussianDiffusion (DDPM). Uses the scalar stable CCNF
    (Definition 4.9) with the unnormalized auto CFM loss L'_Auto
    (Definition 4.6).

    Key parameters:
        lambda_z:  Eigenvalue for spatial state. Controls pull toward data.
        lambda_tau: Eigenvalue for pseudo-time. Controls τ convergence rate.
        lambda_z / lambda_tau: Controls interpolation curvature (Figure 3).
            = 1.0 → recovers OT-FM (Corollary 4.12)
            > 1.0 → z converges faster than τ (sharper transition)

    Training:
        Sample τ ~ U[0,1], z₀ ~ N(0,I), z₁ from data.
        Interpolant (Lemma 4.11):
            z_τ = z₁ + (1-τ)^(λ_z/λ_τ) · (z₀ - z₁)
        Target vector field (Lemma 4.10):
            v'_z(z_τ | z₁) = -λ_z · (z_τ - z₁)
        Loss:
            ||v_θ(z_τ, τ) - v'_z||²

    Sampling:
        Integrate ODE from (z ~ N(0,I), τ=0):
            dz/dt = v_θ(z, τ)
            dτ/dt = λ_τ · (1 - τ)    [analytical: τ(t) = 1 - exp(-λ_τ·t)]
    """

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
        self.vector_field_fn = vector_field_fn
        self.spatial_size = spatial_size
        self.channels = channels
        self.lambda_z = lambda_z
        self.lambda_tau = lambda_tau
        self.ratio = lambda_z / lambda_tau  # λ_z / λ_τ
        self.num_sample_steps = num_sample_steps
        self.loss_type = loss_type

    def interpolate(self, z1, tau):
        """
        Sample from the scalar stable CCNF conditional PDF (Lemma 4.11).

        Given clean data z₁ (= z') and pseudo-time τ, produce:
            z_τ = z₁ + (1-τ)^(λ_z/λ_τ) · (z₀ - z₁)
        where z₀ ~ N(0, I).

        Args:
            z1: (B, C, H, W) clean data samples
            tau: (B,) pseudo-time values in [0, 1]
        Returns:
            z_tau: (B, C, H, W) interpolated noisy samples
            z0: (B, C, H, W) the sampled noise
        """
        z0 = torch.randn_like(z1)
        # alpha = (1 - τ)^(λ_z/λ_τ) — Eq. 35 with τ0=0, τ1=1
        alpha = (1.0 - tau).pow(self.ratio).view(-1, 1, 1, 1)
        z_tau = z1 + alpha * (z0 - z1)
        return z_tau, z0

    def target_vector_field(self, z_tau, z1):
        """
        Target CCNF vector field for z (Lemma 4.10, Eq. 29):
            v'_z(z_τ | z₁) = -λ_z · (z_τ - z₁)

        This is the gradient of H'(z|z') = ½λ_z·||z - z'||² (Def 4.8).
        """
        return -self.lambda_z * (z_tau - z1)

    def forward(self, z1):
        """
        Compute the unnormalized auto CFM loss L'_Auto (Definition 4.6).

        Args:
            z1: (B, C, H, W) batch of clean data
        Returns:
            loss: scalar
        """
        b = z1.shape[0]
        device = z1.device

        # Sample τ ~ U[0, 1]
        tau = torch.rand(b, device=device)

        # Get interpolated sample and target
        z_tau, _ = self.interpolate(z1, tau)
        target = self.target_vector_field(z_tau, z1)

        # Predict vector field
        predicted = self.vector_field_fn(z_tau, tau)

        # Loss
        if self.loss_type == "l1":
            loss = (predicted - target).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(predicted, target)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, num_steps=None):
        """
        Generate samples by integrating the learned ODE.

        System (from Section 4.2):
            dz/dt = v_θ(z, τ)
            dτ/dt = λ_τ · (1 - τ)

        τ has analytical solution: τ(t) = 1 - exp(-λ_τ · t)
        We integrate to T = 5/λ_τ so that τ(T) ≈ 0.993.

        Uses Euler integration. For better quality, increase num_steps.
        """
        num_steps = num_steps or self.num_sample_steps
        device = next(self.vector_field_fn.parameters()).device

        # Initial state: z ~ N(0, I), τ = 0
        S = self.spatial_size
        z = torch.randn(batch_size, self.channels, S, S, device=device)

        # Integration time T s.t. τ(T) ≈ 1
        # τ(T) = 1 - exp(-λ_τ T), so for τ ≈ 0.993: T = 5/λ_τ
        T = 5.0 / self.lambda_tau
        dt = T / num_steps

        # Use analytical τ trajectory for stability
        t_values = torch.linspace(0, T, num_steps + 1, device=device)
        tau_values = 1.0 - torch.exp(-self.lambda_tau * t_values)

        for i in tqdm(range(num_steps), desc="sampling", total=num_steps):
            tau = tau_values[i].expand(batch_size)
            v_z = self.vector_field_fn(z, tau)
            z = z + v_z * dt

        return z

    @torch.no_grad()
    def sample_midpoint(self, batch_size=16, num_steps=None):
        """
        Generate samples using midpoint (RK2) integration for better accuracy.
        """
        num_steps = num_steps or self.num_sample_steps
        device = next(self.vector_field_fn.parameters()).device

        S = self.spatial_size
        z = torch.randn(batch_size, self.channels, S, S, device=device)

        T = 5.0 / self.lambda_tau
        dt = T / num_steps
        t_values = torch.linspace(0, T, num_steps + 1, device=device)
        tau_values = 1.0 - torch.exp(-self.lambda_tau * t_values)

        for i in tqdm(range(num_steps), desc="sampling (midpoint)", total=num_steps):
            tau_i = tau_values[i].expand(batch_size)
            tau_mid = (0.5 * (tau_values[i] + tau_values[i + 1])).expand(batch_size)

            # Half Euler step
            v1 = self.vector_field_fn(z, tau_i)
            z_mid = z + v1 * (dt / 2.0)

            # Full step using midpoint velocity
            v2 = self.vector_field_fn(z_mid, tau_mid)
            z = z + v2 * dt

        return z


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LightcurveDataset(data.Dataset):
    """Dataset that loads pre-processed light curve sequences (list of numpy arrays)."""

    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index], dtype=torch.float32)


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
        super().__init__()
        rank = rank or [0]

        self.model = torch.nn.DataParallel(flow_model, device_ids=rank)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_every = save_every
        self.sample_every = sample_every

        self.batch_size = train_batch_size
        self.spatial_size = spatial_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

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

        self.opt = Adam(flow_model.parameters(), lr=train_lr)
        self.step = 0
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        payload = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        torch.save(payload, str(self.logdir / f"{milestone:08d}-model.pt"))

    def load(self, milestone):
        payload = torch.load(
            str(self.logdir / f"{milestone:08d}-model.pt"), map_location="cpu"
        )
        self.step = payload["step"]
        self.model.load_state_dict(payload["model"])
        self.ema_model.load_state_dict(payload["ema"])

    def train(self):
        t1 = time()
        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl).to(device=DEVICE, dtype=torch.float)
                loss = self.model(batch).sum()
                t0 = time()
                print(f"{self.step}: {loss.item():.6f}  Δt: {t0 - t1:.3f}s")
                t1 = time()
                with open(str(self.logdir / "loss.txt"), "a") as f:
                    f.write(f"{self.step},{loss.item()}\n")
                (loss / self.gradient_accumulate_every).backward()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.sample_every == 0:
                batches = num_to_groups(18, self.batch_size)
                all_images_list = [
                    self.ema_model.module.sample(batch_size=n) for n in batches
                ]
                all_images = torch.cat(all_images_list, dim=0)
                all_images = torch.flip(all_images, dims=[1])
                all_images = [(x - x.min()) / (x.max() - x.min() + 1e-8) for x in all_images]
                utils.save_image(all_images, str(self.logdir / f"{self.step:08d}-sample.jpg"), nrow=6)

            if self.step != 0 and self.step % self.save_every == 0:
                self.save(self.step)

            self.step += 1

        print("Training completed.")
