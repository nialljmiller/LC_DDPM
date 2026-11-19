"""v3 data (docs/DESIGN_v3.md): window rule, representation, per-point features, the training file
written by ``training/build_v3.py`` and padded batches for the two stages.

Stages: ``cadence`` models the sorted log time gaps of a window, p(times | survey, n, span);
``measure`` models (mag, err) at given times, p(mag, err | times, conditions).
"""
import json
from pathlib import Path

import numpy as np
import torch

N_MAX = 1024                  # points per example (and per generated curve)
MIN_POINTS = 10               # examples with fewer points after the window rule are not used
MIN_CYCLES, WINDOW_CYCLES = 1.5, 3.0
N_HARM = 4                    # phase features sin, cos(2 pi k t / P), k = 1..N_HARM
N_FOURIER = 10                # time features sin, cos(pi 2^j x), j < N_FOURIER, x in [0, 1]
GAP_FLOOR = 1e-9              # relative gaps (dt / S) are floored here before the log
PAD_TO = 8                    # batch length is a multiple of this
STAGES = ("cadence", "measure")
NUM_NAMES = {"measure": ("log10_period", "log10_amplitude", "mag", "log10_span"),
             "cadence": ("log10_n", "log10_span")}
CHANNELS = {"measure": 2, "cadence": 1}


# ---------------------------------------------------------------- window rule
def window(t, period, rng, n_max=N_MAX):
    """Indices of the points of one training example and the rule used (1, 2 or 3), or None.

    t: sorted times. Rule 4 first: a curve spanning less than 1.5 periods is not used. Rule 1: n <= n_max,
    whole curve. Rule 2: n_max consecutive points from a random start, if they span >= 3 periods.
    Rule 3: a window of width min(3 P, span) starting at a random observation time (so it is never
    empty), randomly thinned to n_max points when it holds more.
    """
    n, span = len(t), t[-1] - t[0] if len(t) else 0.0
    if n < 2 or span < MIN_CYCLES * period:
        return None
    if n <= n_max:
        return np.arange(n), 1
    s = int(rng.integers(0, n - n_max + 1))
    if t[s + n_max - 1] - t[s] >= WINDOW_CYCLES * period:
        return np.arange(s, s + n_max), 2
    w = min(WINDOW_CYCLES * period, span)
    starts = np.flatnonzero(t - t[0] <= span - w)
    a = int(starts[rng.integers(len(starts))])
    idx = np.arange(a, np.searchsorted(t, t[a] + w, side="right"))
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    return idx, 3


# ---------------------------------------------------------------- representation
def measure_channels(m, e, amplitude):
    """(n, 2) [(m - median m) / A, log10(e / A)]."""
    return np.stack([(m - np.median(m)) / amplitude, np.log10(e / amplitude)], 1)


def measure_values(x, mag, amplitude):
    """Inverse of measure_channels: (mag + A x0, A 10^x1)."""
    return mag + amplitude * x[:, 0], amplitude * 10.0 ** x[:, 1]


def log_gaps(t):
    """log10 of the sorted time gaps relative to the span, floored at GAP_FLOOR."""
    g = np.diff(t) / max(t[-1] - t[0], 1e-300)
    return np.log10(np.maximum(g, GAP_FLOOR))


def times_from_log_gaps(lg, span):
    """Times from log relative gaps, rescaled so that they start at 0 and span exactly ``span``."""
    c = np.concatenate([[0.0], np.cumsum(10.0 ** np.asarray(lg, np.float64))])
    return span * c / c[-1]


def fourier(x, k):
    """sin, cos(pi 2^j x), j < k: (n, 2k)."""
    a = np.pi * np.asarray(x, np.float64)[:, None] * 2.0 ** np.arange(k)
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def measure_features(t, period, gap_norm):
    """Per-point inputs of the measurement model, float32 (n, 1 + 2 N_FOURIER + 2 + 2 N_HARM):
    x = t / S, its Fourier features, standardised log gaps to the previous and next point, and the
    phase features sin, cos(2 pi k t / P) (last 2 N_HARM columns; computed in float64)."""
    t = np.asarray(t, np.float64) - t[0]
    x = t / max(t[-1], 1e-300)
    lg = (log_gaps(t) - gap_norm[0]) / gap_norm[1] if len(t) > 1 else np.zeros(0)
    prev, nxt = np.concatenate([[0.0], lg]), np.concatenate([lg, [0.0]])
    ph = 2 * np.pi * np.mod(t / period, 1.0)[:, None] * np.arange(1, N_HARM + 1)
    return np.concatenate([x[:, None], fourier(x, N_FOURIER), prev[:, None], nxt[:, None], np.sin(ph), np.cos(ph)],
                          1).astype(np.float32)


def cadence_features(n_gaps, n_max=N_MAX):
    """Per-gap inputs of the cadence model (index i): Fourier features of i / n_max (enough to
    resolve single indices) and i / (n_gaps - 1)."""
    i = np.arange(n_gaps, dtype=np.float64)
    return np.concatenate([fourier(i / n_max, int(np.ceil(np.log2(n_max))) + 1),
                           (i / max(n_gaps - 1, 1))[:, None]], 1).astype(np.float32)


def feat_dim(stage, n_max=N_MAX):
    return measure_features(np.arange(3.0), 1.0, (0, 1)).shape[1] if stage == "measure" else \
        cadence_features(2, n_max).shape[1]


PHASE_GATE = (2, feat_dim("measure") - 2 * N_HARM, feat_dim("measure"))  # mask column of log10 P, phase columns


# ---------------------------------------------------------------- training file
def load_v3(data_dir):
    """Arrays of ``train_v3.npz`` and ``vocab.json`` (format: docs/DESIGN_v3.md)."""
    data_dir = Path(data_dir)
    with np.load(data_dir / "train_v3.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads((data_dir / "vocab.json").read_text())


def example(arrays, r):
    """(t, m, e) of example ``r``; t in days from the window's first point."""
    a, b = int(arrays["offsets"][r]), int(arrays["offsets"][r + 1])
    return arrays["time"][a:b], arrays["obs_mag"][a:b].astype(np.float64), arrays["obs_err"][a:b].astype(np.float64)


def raw_numeric(stage, period=None, amplitude=None, mag=None, span=None, n=None):
    """(N, K) numeric conditions of a stage in model order; NaN = missing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if stage == "measure":
            cols = [np.log10(period), np.log10(amplitude), mag, np.log10(span)]
        else:
            cols = [np.log10(n), np.log10(span)]
    return np.stack([np.asarray(c, np.float64) for c in cols], 1)


def example_numeric(arrays, rows, stage):
    a = {k: arrays[k][rows] for k in ("period", "amplitude", "mag", "span", "n_points")}
    return raw_numeric(stage, a["period"], a["amplitude"], a["mag"], a["span"], a["n_points"])


def compute_norm_v3(arrays, rows, stage, max_rows=20000, seed=0):
    """Mean/std of the numeric conditions, the target channels and the log relative gaps (training rows)."""
    num = example_numeric(arrays, rows, stage)
    sub = np.random.default_rng(seed).choice(rows, min(len(rows), max_rows), replace=False)
    gaps, chans = [], []
    for r in sub:
        t, m, e = example(arrays, r)
        gaps.append(log_gaps(t))
        if stage == "measure":
            chans.append(measure_channels(m, e, arrays["amplitude"][r]))
    gaps = np.concatenate(gaps)
    chans = np.concatenate(chans) if chans else gaps[:, None]
    ms = lambda x: ([float(np.nanmean(c)) if np.isfinite(c).any() else 0.0 for c in x.T],
                    [float(np.nanstd(c)) if np.isfinite(c).sum() > 1 and np.nanstd(c) > 0 else 1.0 for c in x.T])
    (mean, std), (cm, cs) = ms(num), ms(chans)
    return {"stage": stage, "names": list(NUM_NAMES[stage]), "mean": mean, "std": std, "chan_mean": cm,
            "chan_std": cs, "gap_mean": float(gaps.mean()), "gap_std": max(float(gaps.std()), 1e-6)}


def encode_v3(survey, cls, num, norm, device="cpu"):
    """Condition dict (survey, cls, num, mask) for raw numeric conditions ``num`` (N, K); survey/cls
    int arrays with negative = missing (cls is always missing for the cadence stage), NaN = missing."""
    survey, cls = np.asarray(survey, np.int64), np.asarray(cls, np.int64)
    if norm["stage"] == "cadence":
        cls = np.full_like(cls, -1)
    x = (np.asarray(num, np.float64) - np.asarray(norm["mean"])) / np.asarray(norm["std"])
    mask = np.concatenate([np.stack([survey >= 0, cls >= 0], 1), np.isfinite(x)], 1)
    return {"survey": torch.as_tensor(np.maximum(survey, 0), device=device),
            "cls": torch.as_tensor(np.maximum(cls, 0), device=device),
            "num": torch.as_tensor(np.nan_to_num(x), dtype=torch.float32, device=device),
            "mask": torch.as_tensor(mask, device=device)}


def pad_stack(seqs, width, length=None):
    """Stack (n_i, width) arrays into (B, L, width) zeros-padded float32 and the (B, L) validity mask."""
    L = length or max(PAD_TO, -(-max(len(s) for s in seqs) // PAD_TO) * PAD_TO)
    out, pad = np.zeros((len(seqs), L, width), np.float32), np.zeros((len(seqs), L), bool)
    for i, s in enumerate(seqs):
        out[i, :len(s)], pad[i, :len(s)] = s, True
    return out, pad


def point_inputs(stage, times, period, norm, n_max=N_MAX):
    """Per-point (measure) or per-gap (cadence) inputs for a list of time arrays."""
    if stage == "measure":
        return [measure_features(t, p, (norm["gap_mean"], norm["gap_std"])) for t, p in zip(times, period)]
    return [cadence_features(len(t) - 1, n_max) for t in times]


def targets(stage, t, m, e, amplitude, norm):
    """Standardised model targets of one example: (n, 2) for measure, (n - 1, 1) for cadence."""
    if stage == "measure":
        x = measure_channels(m, e, amplitude)
        return (x - np.asarray(norm["chan_mean"])) / np.asarray(norm["chan_std"])
    return ((log_gaps(t) - norm["gap_mean"]) / norm["gap_std"])[:, None]


def make_batch(arrays, rows, stage, norm, device="cpu", n_max=N_MAX):
    """(x0 (B, L, C), condition dict with feat and pad) for training rows of the v3 file."""
    ex = [example(arrays, r) for r in rows]
    x0, pad = pad_stack([targets(stage, t, m, e, arrays["amplitude"][r], norm) for (t, m, e), r in zip(ex, rows)],
                        CHANNELS[stage])
    feat, _ = pad_stack(point_inputs(stage, [x[0] for x in ex], arrays["period"][rows], norm, n_max),
                        feat_dim(stage, n_max), x0.shape[1])
    cond = encode_v3(arrays["survey"][rows], arrays["cls"][rows], example_numeric(arrays, rows, stage), norm, device)
    cond.update(feat=torch.as_tensor(feat, device=device), pad=torch.as_tensor(pad, device=device))
    return torch.as_tensor(x0, device=device), cond
