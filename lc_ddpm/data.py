"""Profile bank (``profiles.npz`` + ``vocab.json``, see docs/DESIGN.md) and condition encoding."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NUMERIC = ("log10_period", "log10_amplitude", "mag")


def load_bank(bank_dir):
    """Return (arrays dict from profiles.npz, vocab dict)."""
    bank_dir = Path(bank_dir)
    with np.load(bank_dir / "profiles.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    vocab = json.loads((bank_dir / "vocab.json").read_text())
    return arrays, vocab


def raw_numeric(period, amplitude, mag):
    """(N, 3) float64 [log10 P, log10 A, mag]; NaN marks a missing value."""
    period, amplitude, mag = (np.asarray(v, dtype=np.float64) for v in (period, amplitude, mag))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.log10(period), np.log10(amplitude), mag], 1)


def compute_norm(arrays):
    """Mean and std of each numeric condition over finite values."""
    x = raw_numeric(arrays["period"], arrays["amplitude"], arrays["mag"])
    mean = [float(np.nanmean(c)) if np.isfinite(c).any() else 0.0 for c in x.T]
    std = [float(np.nanstd(c)) if np.isfinite(c).sum() > 1 else 1.0 for c in x.T]
    return {"names": list(NUMERIC), "mean": mean, "std": [s if s > 0 else 1.0 for s in std]}


def encode_conditions(survey, cls, period, amplitude, mag, norm, device="cpu"):
    """Condition dict for the model. survey/cls: int arrays, negative = missing;
    period/amplitude/mag: arrays, NaN = missing."""
    survey, cls = np.asarray(survey, np.int64), np.asarray(cls, np.int64)
    x = (raw_numeric(period, amplitude, mag) - np.asarray(norm["mean"])) / np.asarray(norm["std"])
    mask = np.concatenate([np.stack([survey >= 0, cls >= 0], 1), np.isfinite(x)], 1)
    return {"survey": torch.as_tensor(np.maximum(survey, 0), device=device),
            "cls": torch.as_tensor(np.maximum(cls, 0), device=device),
            "num": torch.as_tensor(np.nan_to_num(x), dtype=torch.float32, device=device),
            "mask": torch.as_tensor(mask, device=device)}


def select_rows(arrays, min_snr=None, tiers=None):
    """Indices of rows passing the snr / tier filters."""
    keep = np.ones(len(arrays["profile"]), bool)
    if min_snr is not None:
        keep &= arrays["snr"] >= min_snr
    if tiers is not None:
        keep &= np.isin(arrays["tier"], tiers)
    return np.flatnonzero(keep)


def row_weights(arrays, rows, tier_weight=1.0):
    """Per-row sampling/loss weight: 1 for tier 0, ``tier_weight`` for tier 1."""
    return np.where(arrays["tier"][rows] == 1, tier_weight, 1.0).astype(np.float32)


class ProfileDataset(Dataset):
    """Items: {"x": (128,) in [-1, 1], "survey", "cls", "num": (3,), "mask": (5,), "weight"}.

    The bank holds only training-split stars. No augmentation: index 0 is the faintest
    point by convention, so circular rolls or mirroring would change the meaning.
    """

    def __init__(self, bank_dir, min_snr=None, tiers=None, tier_weight=1.0, norm=None, rows=None):
        self.arrays, self.vocab = load_bank(bank_dir)
        self.norm = norm or compute_norm(self.arrays)
        self.rows = select_rows(self.arrays, min_snr, tiers) if rows is None else np.asarray(rows)
        a = self.arrays
        self.x = torch.as_tensor(2 * a["profile"][self.rows] - 1, dtype=torch.float32)
        self.cond = encode_conditions(a["survey"][self.rows], a["cls"][self.rows], a["period"][self.rows],
                                      a["amplitude"][self.rows], a["mag"][self.rows], self.norm)
        self.weight = torch.as_tensor(row_weights(a, self.rows, tier_weight))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"x": self.x[i], **{k: v[i] for k, v in self.cond.items()}, "weight": self.weight[i]}
