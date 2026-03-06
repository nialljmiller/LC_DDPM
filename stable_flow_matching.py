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


from primvs_pipeline import primvs_api as api
PrimvsCatalog = api.PrimvsCatalog


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


def _delete_rand_items_3col(a, b, c, n):
    keep = np.sort(np.random.choice(len(a), len(a) - n, replace=False))
    return a[keep], b[keep], c[keep]


def tile_to_square(arr, spatial_size):
    n = spatial_size * spatial_size
    reps = math.ceil(n / len(arr))
    return np.tile(arr, reps)[:n].reshape(spatial_size, spatial_size)


class LazyPrimvsDataset(data.Dataset):
    """
    Reads each CSV from disk on demand in __getitem__.
    RAM usage is O(batch_size), not O(dataset_size).
    """
    def __init__(self, source_ids, data_dir, lc_size=70, spatial_size=128, band="Ks"):
        super().__init__()
        self.cat = PrimvsCatalog(data_dir)
        self.lc_size = lc_size
        self.spatial_size = spatial_size
        self.band = band
        # Only keep IDs whose CSV exists on disk — checked once at startup.
        self.source_ids = [int(sid) for sid in source_ids if self.cat.source_exists(sid)]
        print(f"LazyPrimvsDataset: {len(self.source_ids)} sources on disk "
              f"(band={band}, lc_size={lc_size}, spatial={spatial_size}x{spatial_size})")

    def __len__(self):
        return len(self.source_ids)

    def _process(self, sid):
        df = self.cat.get_lightcurve(sid)
        if df is None:
            return None
        subset = df[df["filter"] == self.band].dropna(subset=["mag", "err"]).copy()
        mjd = subset["mjd"].values.astype(np.float32)
        mag = subset["mag"].values.astype(np.float32)
        err = subset["err"].values.astype(np.float32)

        finite = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(err)
        positive_err = err > 0
        keep = finite & positive_err
        if keep.sum() < self.lc_size:
            return None
        mjd, mag, err = mjd[keep], mag[keep], err[keep]

        order = np.argsort(mjd)
        mjd, mag, err = mjd[order], mag[order], err[order]
        if len(mag) > self.lc_size:
            mjd, mag, err = _delete_rand_items_3col(mjd, mag, err, len(mag) - self.lc_size)

        # Robust normalization keeps values in a stable dynamic range.
        mag_median = np.median(mag)
        mag_scale = np.median(np.abs(mag - mag_median)) * 1.4826
        if not np.isfinite(mag_scale) or mag_scale < 1e-3:
            mag_scale = np.std(mag)
        if not np.isfinite(mag_scale) or mag_scale < 1e-3:
            return None

        mag = np.clip((mag - mag_median) / (mag_scale + 1e-6), -8.0, 8.0)
        err = np.clip(err / (mag_scale + 1e-6), 0.0, 4.0)

        mjd_range = mjd.max() - mjd.min()
        if mjd_range == 0:
            return None
        phase = (mjd - mjd.min()) / mjd_range

        if not (np.all(np.isfinite(mag)) and np.all(np.isfinite(err)) and np.all(np.isfinite(phase))):
            return None

        s = self.spatial_size
        seq = np.stack([
            tile_to_square(mag, s),
            tile_to_square(err, s),
            tile_to_square(phase, s),
        ], axis=0).astype(np.float32)
        return seq

    def __getitem__(self, index):
        seq = self._process(self.source_ids[index])
        if seq is None:
            seq = self._process(self.source_ids[(index + 1) % len(self.source_ids)])
        if seq is None:
            seq = np.zeros((3, self.spatial_size, self.spatial_size), dtype=np.float32)
        return torch.from_numpy(seq)


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
        self.ratio = lambda_z / lambda_tau
        self.num_sample_steps = num_sample_steps
        self.loss_type = loss_type

    def interpolate(self, z1, tau):
        z0 = torch.randn_like(z1)
        alpha = (1.0 - tau).pow(self.ratio).view(-1, 1, 1, 1)
        z_tau = z1 + alpha * (z0 - z1)
        return z_tau, z0

    def target_vector_field(self, z_tau, z1):
        return -self.lambda_z * (z_tau - z1)

    def forward(self, z1):
        b = z1.shape[0]
        device = z1.device
        tau = torch.rand(b, device=device)
        z_tau, _ = self.interpolate(z1, tau)
        target = self.target_vector_field(z_tau, z1)
        predicted = self.vector_field_fn(z_tau, tau)
        if self.loss_type == "l1":
            loss = (predicted - target).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(predicted, target)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, num_steps=None):
        num_steps = num_steps or self.num_sample_steps
        device = next(self.vector_field_fn.parameters()).device
        S = self.spatial_size
        z = torch.randn(batch_size, self.channels, S, S, device=device)
        T = 5.0 / self.lambda_tau
        dt = T / num_steps
        t_values = torch.linspace(0, T, num_steps + 1, device=device)
        tau_values = 1.0 - torch.exp(-self.lambda_tau * t_values)
        for i in tqdm(range(num_steps), desc="sampling", total=num_steps):
            tau = tau_values[i].expand(batch_size)
            v_z = self.vector_field_fn(z, tau)
            z = z + v_z * dt
        return z

    @torch.no_grad()
    def sample_midpoint(self, batch_size=16, num_steps=None):
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
            v1 = self.vector_field_fn(z, tau_i)
            z_mid = z + v1 * (dt / 2.0)
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
        grad_clip_norm=1.0,
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
        self.grad_clip_norm = grad_clip_norm

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
                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss at step {self.step}; skipping optimizer update")
                    self.opt.zero_grad(set_to_none=True)
                    break
                (loss / self.gradient_accumulate_every).backward()
            else:
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

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
